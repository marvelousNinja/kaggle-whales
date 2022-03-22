from datetime import datetime
from functools import partial
import math
import os
import random
import json

import albumentations
import cv2; cv2.setNumThreads(0)
import faiss
from fire import Fire
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
import torch.utils.data.dataloader
from torch.utils.data import Dataset, DataLoader

from fyx.callbacks.callback import Callback
from fyx.callbacks.cyclic_lr import CyclicLR
from fyx.callbacks.model_checkpoint import ModelCheckpoint, load_checkpoint
from fyx.callbacks.tensorboard_monitor import TensorboardMonitor
from fyx.callbacks.lr_schedule import LRSchedule
from fyx.loggers import make_loggers
from fyx.training import fit_model
from fyx.utils import as_cuda

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def collate_fn(batch):
    batch = [el for nanobatch in batch for el in nanobatch]
    return torch.utils.data.dataloader.default_collate(batch)

class TransformedDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __getitem__(self, item):
        return self.transform(self.records.iloc[item])

    def __len__(self):
        return len(self.records)

class ArcfaceHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(features), torch.nn.functional.normalize(self.weight))
        return cosine

def pairwise_cosface(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        gamma: float, ) -> torch.Tensor:
    # Normalize embedding features
    embedding = torch.nn.functional.normalize(embedding, dim=1)

    dist_mat = torch.matmul(embedding, embedding.t())

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    # Mask scores related to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    logit_p = -gamma * s_p + (-99999999.) * (1 - is_pos)
    logit_n = gamma * (s_n + margin) + (-99999999.) * (1 - is_neg)

    loss = torch.nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss

def arcface_loss(logits, labels, s=30.0, m=0.50):
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m

    cosine = logits
    sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).to(logits)
    phi = cosine * cos_m - sine * sin_m
    phi = torch.where(cosine > th, phi, cosine - mm)

    labels2 = torch.zeros_like(cosine)
    labels2.scatter_(1, labels.view(-1, 1).long(), 1)
    output = (labels2 * phi) + ((1.0 - labels2) * cosine)

    output = output * s
    return torch.nn.functional.cross_entropy(output, labels)

def simpler_arcface_loss(logits, labels, m=0.5, s=30.0):
    index = torch.where(labels != -1)[0]
    m_hot = torch.zeros(index.size()[0], logits.size()[1], device=logits.device)
    m_hot.scatter_(1, labels[index, None], m)
    logits.acos_()
    logits[index] += m_hot
    logits.cos_().mul_(s)
    return torch.nn.functional.cross_entropy(logits, labels)

def gem(x, p=3, eps=1e-6):
    return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = torch.nn.Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

class WhaleNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features, logits = self.model(x['image'])
        return {
            'features': features,
            'individual_logits': logits
        }

class InnerNet(torch.nn.Module):
    def __init__(self, input_shape, encoder_name, encoder_weights, num_identities, global_pool, neck, classifier, features_before_neck):
        super().__init__()
        self.model = timm.create_model(encoder_name, pretrained=encoder_weights is not None)
        self.input_shape = input_shape

        self.global_pool = {
            'avg_pool': lambda: torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            ),
            'gem': lambda: torch.nn.Sequential(
                GeM(),
                torch.nn.Flatten()
            ),
            'fc': lambda: torch.nn.Sequential(
                torch.nn.BatchNorm2d(self.model.num_features),
                torch.nn.Dropout(0.3),
                torch.nn.Flatten(),
                torch.nn.LazyLinear(512),
                torch.nn.BatchNorm1d(512)
            )
        }[global_pool]()

        self.neck = {
            'identity': lambda: torch.nn.Identity(),
            'fc_bn': lambda: torch.nn.Sequential(
                torch.nn.Linear(self.model.num_features, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.PReLU()
            )
        }[neck]()

        emgedding_size = self.model.num_features if neck == 'identity' and global_pool != 'fc' else 512

        self.classifier = {
            'arcface': lambda: ArcfaceHead(emgedding_size, num_identities),
            'fc': lambda: torch.nn.Linear(emgedding_size, num_identities)
        }[classifier]()

        self.features_before_neck = features_before_neck

    def forward(self, x):
        pooled_features = self.global_pool(self.model.forward_features(x))
        features = self.neck(pooled_features)
        logits = self.classifier(features)

        if self.features_before_neck:
            features = pooled_features

        return features, logits

    def sample_input(self):
        return torch.randint(255, size=(1, 3, *self.input_shape)).float().to(next(self.parameters()).device)

def convert_to_rgb(tensor):
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = image[:, :, [0, 0, 0]] if image.shape[2] == 1 else image[:, :, [2, 1, 0]]
    return np.ascontiguousarray(image)

def visualize_preds(writer, tag, outputs, batch, counter):
    features = outputs['features']
    similarity_matrix = torch.nn.functional.cosine_similarity(features[:,:,None], features.t()[None,:,:])
    similarity_matrix.fill_diagonal_(-1.0)
    top_similarity, top_matches = similarity_matrix.sort(descending=True, dim=-1)

    for i in range(min(len(batch['image']), 16)):
        image = convert_to_rgb(batch['image'][i])
        top_match_indices = top_matches[i][:5]
        top_match_images = [convert_to_rgb(batch['image'][index]) for index in top_match_indices]
        top_match_identities = [batch['individual_label'][index].item() for index in top_match_indices]

        for match_idx, matched_image in enumerate(top_match_images):
            if batch['individual_label'][i].item() == top_match_identities[match_idx]:
                color = [0.0, 1.0, 0.0]
            else:
                color = [1.0, 0.0, 0.0]

            inner_mask = np.zeros(matched_image.shape[:-1], dtype=bool)
            inner_mask[10:-10, 10:-10] = True
            matched_image[~inner_mask] = color

        combined = (np.hstack([image, *top_match_images]) * 255).astype(np.uint8)

        if tag == 'val':
            prefix = f'{tag}/{batch["image_name"][i]}'
            writer.add_image(f'{prefix}', combined, dataformats='HWC', global_step=counter)
        else:
            prefix = f'{tag}/{i}'
            writer.add_image(f'{prefix}', combined, dataformats='HWC', global_step=counter)


# TODO AS: Triplet loss copied from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/layers/triplet_loss.py
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    return torch.cdist(x, y)

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

def compute_loss(outputs, batch, loss_weights):
    total_loss = 0.0
    losses = {
        'arcface': lambda: arcface_loss(outputs['individual_logits'], batch['individual_label']),
        'ce': lambda: torch.nn.functional.cross_entropy(outputs['individual_logits'], batch['individual_label'], label_smoothing=0.1),
        'triplet': lambda: TripletLoss(margin=0.3)(outputs['features'], batch['individual_label'])[0],
        'pairwise_cosface': lambda: pairwise_cosface(outputs['features'], batch['individual_label'], margin=0.5, gamma=30.0),
        'simplified_arcface': lambda: simpler_arcface_loss(outputs['individual_logits'], batch['individual_label'])
    }

    loss_values = {}

    for name, weight in loss_weights.items():
        loss_values[name] = losses[name]()
        total_loss += loss_values[name] * weight

    return {'total_loss': total_loss, **loss_values}

def transform(input_shape, augment, debug, mapping, num_images_per_identity, record):
    if debug:
        import pdb; pdb.set_trace()

    records = []

    if augment:
        image_paths = np.random.choice(record['image_paths'], size=num_images_per_identity, replace=True)
    else:
        # TODO AS: Make this deterministic
        image_paths = np.random.choice(record['image_paths'], size=num_images_per_identity, replace=True)
        # image_paths = record['image_paths'][:num_images_per_identity]

    for image_path in image_paths:
        image = cv2.imread(image_path)

        if augment:
            steps = [
                albumentations.Resize(input_shape[0], input_shape[1]),
                albumentations.PadIfNeeded(input_shape[0] + 20, input_shape[1] + 20),
                albumentations.RandomCrop(height=input_shape[0], width=input_shape[1], p=1.0),
                albumentations.HorizontalFlip(p=0.5)
            ]
        else:
            steps = [
                albumentations.Resize(input_shape[0], input_shape[1])
            ]

        individual_image = albumentations.Compose(steps)(image=image)['image']

        records.append({
            'image': np.moveaxis(individual_image, -1, 0).astype(np.float32) / 255.0,
            'image_name': os.path.basename(image_path),
            'image_path': image_path,
            'individual_label': mapping[record['individual_id']],
            'individual_id': record['individual_id']
        })

    return records

class MeanAveragePrecision(Callback):
    def __init__(self):
        pass

    def on_epoch_begin(self, logs):
        self.total = 0
        self.num_records = 0
        self.individual_ids = []
        self.features = []
        self.query_individual_ids = []
        self.query_features = []
        self.index = None

    def on_train_batch_end(self, logs, outputs, batch):
        self.individual_ids.extend(batch['individual_id'])
        self.features.extend(torch.nn.functional.normalize(outputs['features'].detach()).cpu().numpy())

    def on_validation_batch_end(self, logs, outputs, batch):
        if not type(self.features) is np.ndarray:
            self.features = np.stack(self.features)
            self.individual_ids = np.array(self.individual_ids)
            self.index = faiss.IndexFlatL2(self.features.shape[-1])
            self.index.add(self.features.astype(np.float32))
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)


        query_individual_ids = np.array(batch['individual_id'])
        query_features = torch.nn.functional.normalize(outputs['features']).cpu().numpy().astype(np.float32)

        _, top_matches = self.index.search(query_features, k=5)

        for i in range(len(top_matches)):
            # TODO AS: Keep only unique matches
            # list(dict.fromkeys(top_matches[i]))[:5]
            record_matches = top_matches[i][:5]
            mapped_record_matches = self.individual_ids[record_matches]
            self.num_records += 1

            for j in range(5):
                if mapped_record_matches[j] == query_individual_ids[i]:
                    self.total += (1.0 / (j + 1))
                    break

    def on_epoch_end(self, logs):
        logs['val_mAP'] = self.total / self.num_records
        self.index = None

def fit(
    name='default',
    num_epochs=200,
    encoder_name='resnet50',
    encoder_weights='imagenet',
    sampling_strategy='random',
    input_shape='256x128',
    val_input_shape=None,
    optimizer_name='adam',
    limit=None,
    validation_limit=None,
    batch_size=None,
    validation_batch_size=None,
    lr=0.00035,
    checkpoint_path=None,
    num_folds=10,
    train_fold_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    validation_fold_ids=[9],
    steps_per_epoch=1000,
    seed=42,
    cyclic_lr=False,
    accumulate_n_batches=1,
    mixed_precision=True,
    debug=False,
    profile=False,
    num_identities=16,
    num_images_per_identity=4,
    use_phalanx_dataset=False,
    neck='identity',
    classifier='fc',
    global_pool='avg_pool',
    lr_schedule='regular',
    features_before_neck=False,
    loss_weights={'ce': 1.0, 'triplet': 1.0},
    clip_gradient_norm_to=5.0,
    **kwargs
):
    assert len(kwargs) == 0, f'Unrecognized args: {kwargs}'
    args_to_save = dict(locals())

    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    logger, _ = make_loggers(False)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    experiment_path = f'data/experiments/{timestamp}_{name}'

    os.makedirs(experiment_path, exist_ok=True)
    with open(f'{experiment_path}/params.json', 'w') as f:
        json.dump(args_to_save, f, indent=2)

    input_shape = tuple(map(int, input_shape.split('x')))
    val_input_shape = tuple(map(int, val_input_shape.split('x'))) if val_input_shape is not None else input_shape

    batch_size = batch_size if batch_size is not None else num_identities
    assert batch_size == num_identities

    validation_batch_size = batch_size if validation_batch_size is None else validation_batch_size

    df = pd.read_csv('data/train.csv')

    if checkpoint_path:
        model = WhaleNet(load_checkpoint(checkpoint_path))
    else:
        model = WhaleNet(InnerNet(
            input_shape, encoder_name, encoder_weights,
            num_identities=len(df['individual_id'].unique()), global_pool=global_pool, neck=neck, classifier=classifier,
            features_before_neck=features_before_neck
        ))

    model = as_cuda(model)

    optimizer = {
        'adam': lambda: torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr),
        'sgd': lambda: torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, weight_decay=1e-4, momentum=0.9, nesterov=True),
        'adamw': lambda: torch.optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr),
    }[optimizer_name]()

    if use_phalanx_dataset:
        df['image_path'] = './data/cropped_train_images/cropped_train_images/' + df['image']
    else:
        df['image_path'] = './data/train_images/' + df['image']

    df['species'] = df['species'].replace({
        'globis': 'short_finned_pilot_whale',
        'pilot_whale': 'short_finned_pilot_whale',
        'kiler_whale': 'killer_whale',
        'bottlenose_dolpin': 'bottlenose_dolphin'
    })

    df['image_count'] = df['individual_id'].map(df.groupby('individual_id')['image'].count())
    label_mapping = dict(zip(df['individual_id'].unique(), np.arange(len(df['individual_id'].unique()))))

    # I.e. the same identity can span multiple folds
    df['fold_id'] = np.random.randint(0, num_folds, len(df))
    df['weight'] = 1.0

    # TODO AS: What is the ratio of "unlabeled" identities? Probe the LB? We can try to tweak it.
    train_df = df[df['fold_id'].isin(train_fold_ids)].copy()
    train_df['image_paths'] = train_df['individual_id'].map(train_df.groupby('individual_id')['image_path'].apply(list))
    train_df = train_df.drop_duplicates('individual_id')

    val_df = df[df['fold_id'].isin(validation_fold_ids)].copy()
    val_df['image_paths'] = val_df['individual_id'].map(val_df.groupby('individual_id')['image_path'].apply(list))
    val_df = val_df.drop_duplicates('individual_id')

    train_dataset = TransformedDataset(train_df[:limit], partial(transform, input_shape, True, debug, label_mapping, num_images_per_identity))
    validation_dataset = TransformedDataset(val_df, partial(transform, val_input_shape, False, debug, label_mapping, num_images_per_identity))

    num_workers = 16 if torch.cuda.is_available() and not debug else 0

    # TODO AS: `validation_limit` strictly selects subset for all epochs
    # but `limit` limits number of samples per epoch, while preserving the dataset
    # how to perform overfit test?
    train_sampler, validation_sampler = {
        'random': [
            torch.utils.data.RandomSampler(range(len(train_dataset)), replacement=False, num_samples=10000),
            torch.utils.data.RandomSampler(range(len(validation_dataset)))
        ],
        'weighted': [
            torch.utils.data.WeightedRandomSampler(num_samples=10000, weights=train_df['weight'].values),
            torch.utils.data.RandomSampler(range(len(validation_dataset)))
        ]
    }[sampling_strategy]

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=collate_fn
    )

    validation_dataloader = DataLoader(
        torch.utils.data.Subset(validation_dataset, list(validation_sampler)[:validation_limit]),
        batch_size=validation_batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    lr_schedule = {
        'regular': lambda:[(0, lr), (40, lr / 10), (70, lr / 100)],
        'warmup': lambda: [*[(i, (1 + i) * lr / 10) for i in range(10)], (40, lr / 10), (70, lr / 100)]
    }[lr_schedule]()

    callbacks = [
        MeanAveragePrecision(),
        ModelCheckpoint(model.model, experiment_path, 'val_mAP', 'max', logger),
        LRSchedule(optimizer, lr_schedule, logger),
        TensorboardMonitor(experiment_path, visualize_fn=visualize_preds),
    ]

    if cyclic_lr:
       callbacks.append(CyclicLR(optimizer=optimizer, base_lr=lr, max_lr=lr * 6, step_size_up=len(train_dataloader) // 2, step_size_down=(len(train_dataloader) + 1) // 2, cycle_momentum=False))

    fit_model(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_fn=partial(compute_loss, loss_weights=loss_weights),
        num_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        accumulate_n_batches=accumulate_n_batches,
        mixed_precision=mixed_precision,
        profile=profile,
        profile_path=experiment_path + '_profile',
        clip_gradient_norm_to=clip_gradient_norm_to
    )


if __name__ == '__main__':
    Fire(fit)
