import contextlib
from collections import defaultdict
from datetime import datetime
from functools import partial, reduce
from itertools import product

import os
import operator
import random
import json

import albumentations
import cv2; cv2.setNumThreads(0)
from fire import Fire
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise
import timm
import torch
from torch import nn
import torch.utils.data.dataloader
from torch.utils.data import Dataset, DataLoader

from fyx.callbacks.meter import Meter
from fyx.callbacks.cyclic_lr import CyclicLR
from fyx.callbacks.dump import Dump
from fyx.callbacks.model_checkpoint import ModelCheckpoint, load_checkpoint
from fyx.callbacks.tensorboard_monitor import TensorboardMonitor
from fyx.callbacks.lr_schedule import LRSchedule
from fyx.loggers import make_loggers
from fyx.training import fit_model
from fyx.utils import as_cuda, to_numpy

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
    def __init__(self, input_shape, encoder_name, encoder_weights, num_identities):
        super().__init__()
        self.model = timm.create_model(encoder_name, pretrained=encoder_weights is not None)
        self.input_shape = input_shape
        self.fc = torch.nn.Linear(self.model.num_features, num_identities)
        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )

    def forward(self, x):
        features = self.pool(self.model.forward_features(x))
        logits = self.fc(features)
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

def compute_loss(outputs, batch):
    id_loss = torch.nn.functional.cross_entropy(outputs['individual_logits'], batch['individual_label'], label_smoothing=0.1)
    triplet_loss = TripletLoss(margin=0.3)(outputs['features'], batch['individual_label'])[0]
    total_loss = id_loss + triplet_loss
    return {'total_loss': total_loss, 'id_loss': id_loss, 'triplet_loss': triplet_loss}


def transform(input_shape, augment, debug, mapping, num_images_per_identity, record):
    if debug:
        import pdb; pdb.set_trace()

    records = []

    if augment:
        image_paths = np.random.choice(record['image_paths'], size=num_images_per_identity, replace=True)
    else:
        image_paths = record['image_paths'][:num_images_per_identity]

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

def mean_average_precision(outputs, batch):
    values = []

    for i in range(len(batch['image'])):
        values.append({
            'image_name': batch['image_name'][i],
            'individual_label': batch['individual_label'][i].item(),
            'features': outputs['features'][i].cpu().numpy()
        })

    return values

def reduce_map(records):
    # TODO AS: This is implemented in numpy and duplicates torch stuff in visualise_preds
    df = pd.DataFrame(sum(records, []))
    all_labels = df['individual_label']
    all_features = np.stack(df['features'].values)
    similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(all_features, all_features)
    np.fill_diagonal(similarity_matrix, -1.0)
    top_matches = similarity_matrix.argsort()[:, ::-1][:, :5]
    mapped_matches = all_labels[:, None][top_matches][:, :, 0]

    total = 0
    for i in range(len(top_matches)):
        if all_labels[i] in mapped_matches[i]:
            total += 1.0

    total /= len(top_matches)
    return total

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

    torch.autograd.set_detect_anomaly(debug)

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
        model = WhaleNet(InnerNet(input_shape, encoder_name, encoder_weights, num_identities=len(df['individual_id'].unique())))

    model = as_cuda(model)

    optimizer = {
        'adam': lambda: torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr),
        'sgd': lambda: torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, weight_decay=1e-4, momentum=0.9, nesterov=True),
        'adamw': lambda: torch.optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr),
    }[optimizer_name]()

    df['image_path'] = './data/train_images/' + df['image']
    df['species'] = df['species'].replace({
        'globis': 'short_finned_pilot_whale',
        'pilot_whale': 'short_finned_pilot_whale',
        'kiler_whale': 'killer_whale',
        'bottlenose_dolpin': 'bottlenose_dolphin'
    })

    df['image_count'] = df['individual_id'].map(df.groupby('individual_id')['image'].count())

    fold_mapping = dict(zip(df['individual_id'].unique(), np.random.randint(0, num_folds, len(df['individual_id'].unique()))))
    df['fold_id'] = df['individual_id'].map(fold_mapping)
    label_mapping = dict(zip(df['individual_id'].unique(), np.arange(len(df['individual_id'].unique()))))
    df['image_paths'] = df['individual_id'].map(df.groupby('individual_id')['image_path'].apply(list))

    df = df.drop_duplicates('individual_id').copy()
    df['weight'] = 1.0

    train_df = df[df['fold_id'].isin(train_fold_ids)].sample(frac=1.0)
    val_df = df[df['fold_id'].isin(validation_fold_ids)]
    val_df = val_df[val_df['image_count'] == num_images_per_identity]

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
        num_workers=num_workers // 2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    callbacks = [
        Meter('mAP', map_fn=mean_average_precision, reduce_fn=reduce_map, reduce_once=True, only_val=True),
        ModelCheckpoint(model.model, experiment_path, 'val_mAP', 'max', logger),
        TensorboardMonitor(experiment_path, visualize_fn=visualize_preds),
        # LRSchedule(optimizer, [(0, lr), (15, lr / 5), (50, lr / 10)], logger),
        # Dump(lambda logs: str(logs['val_mAP_all']), 'out.txt')
    ]

    if cyclic_lr:
       callbacks.append(CyclicLR(optimizer=optimizer, base_lr=lr, max_lr=lr * 6, step_size_up=len(train_dataloader) // 2, step_size_down=(len(train_dataloader) + 1) // 2, cycle_momentum=False))

    fit_model(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        accumulate_n_batches=accumulate_n_batches,
        mixed_precision=mixed_precision,
        profile=profile,
        profile_path=experiment_path + '_profile'
    )


if __name__ == '__main__':
    Fire(fit)
