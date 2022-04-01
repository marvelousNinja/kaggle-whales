import os
from functools import partial

import albumentations
import cv2
import faiss
import tqdm
import glob
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import Dataset, DataLoader

def load_checkpoint(path):
    if path.endswith('.serialized'):
        return torch.load(path, map_location='cpu')
    else:
        return torch.jit.load(path, map_location='cpu')

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def from_numpy(obj):

    if isinstance(obj, dict):
        return {key: from_numpy(value) for key, value in obj.items()}

    if torch.cuda.is_available():
        if isinstance(obj, torch.Tensor): return obj.to('cuda')
        return obj
    else:
        if isinstance(obj, torch.Tensor): return obj
        return obj

class TransformedDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __getitem__(self, item):
        return self.transform(self.records.iloc[item])

    def __len__(self):
        return len(self.records)

def transform(input_shape, record):
    image = cv2.imread(record['image_path'])

    steps = [
        albumentations.Resize(input_shape[0], input_shape[1])
    ]

    individual_image = albumentations.Compose(steps)(image=image)['image']

    return {
        'image': np.moveaxis(individual_image, -1, 0).astype(np.float32) / 255.0,
        **record
    }

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


def predict(
    batch_size=64,
    num_workers=16,
    model_path='data/experiments/20220328_195712_arcface_triplet_neck_bn_pool_gem_s_auto_m05/best.pth',
    submission_path='submission.csv',
    input_shape='256x256',
    dataset_name='fullbody',
    limit=None,
    **kwargs
):
    assert len(kwargs) == 0, f'Unrecognized args: {kwargs}'

    input_shape = tuple(map(int, input_shape.split('x')))

    model = WhaleNet(load_checkpoint(model_path))
    model = as_cuda(model)
    model = model.eval()
    torch.set_grad_enabled(False)

    train_df = pd.read_csv('data/train.csv')
    train_df = train_df.rename(columns={'image': 'image_name'})

    if dataset_name == 'default':
        train_df['image_path'] = 'data/train_images/' + train_df['image_name']
        test_image_paths = glob.glob('data/test_images/**.jpg')
    elif dataset_name == 'phalanx':
        train_df['image_path'] = 'data/cropped_train_images/cropped_train_images/' + train_df['image_name']
        test_image_paths = glob.glob('data/cropped_test_images/cropped_test_images/**.jpg')
    elif dataset_name == 'fullbody':
        train_df['image_path'] = 'data/train/' + train_df['image_name']
        test_image_paths = glob.glob('data/test/**.jpg')
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    train_df['image_path'] = 'data/cropped_train_images/cropped_train_images/' + train_df['image_name']
    test_image_paths = glob.glob('data/cropped_test_images/cropped_test_images/**.jpg')

    test_df = pd.DataFrame({
        'image_path': test_image_paths
    })
    test_df['image_name'] = test_df['image_path'].str.split('/').str[-1]
    combined_df = pd.concat([train_df, test_df])

    dataset = TransformedDataset(combined_df, partial(transform, input_shape))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    features = []

    for batch in tqdm.tqdm(dataloader):
        batch = from_numpy(batch)
        outputs = model(batch)

        batch_features = torch.nn.functional.normalize(outputs['features'].detach()).cpu().numpy()
        features.extend(batch_features)

    combined_df['features'] = features

    train_df = combined_df[combined_df['individual_id'].notnull()]
    test_df = combined_df[~combined_df['individual_id'].notnull()]

    index = faiss.IndexFlatL2(outputs['features'].shape[-1])
    index.add(np.stack(train_df['features'].values))
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    all_labels = train_df['individual_id']
    _, top_matches = index.search(np.stack(test_df['features'].values), k=5)
    mapped_matches = all_labels[:, None][top_matches][:, :, 0]
    mapped_matches = list(map(lambda arr: ' '.join(arr), mapped_matches))
    pd.DataFrame({'image': test_df['image_name'], 'predictions': mapped_matches}).to_csv(submission_path, index=False)

if __name__ == '__main__':
    if os.getenv('KAGGLE_CONTAINER_NAME') is not None:
        FILES_PATH = 'placeholder'

        predict(
            batch_size=12,
            num_workers=2,
            dataset_path='/kaggle/input/hubmap-kidney-segmentation/test',
            submission_path='/kaggle/working/submission.csv',
            extra_files_path=FILES_PATH
        )
    else:
        import fire
        fire.Fire(predict)