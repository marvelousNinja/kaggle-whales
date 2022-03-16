import contextlib
import os
import operator
import math
from functools import partial, reduce

import cv2
import tqdm
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def load_checkpoint(path):
    if path.endswith('.serialized'):
        return torch.load(path, map_location='cpu')
    else:
        return torch.jit.load(path, map_location='cpu')


# TODO AS: How to check for out of bounds?
def generate_crop_parameters(image_shape, crop_shape, overlap_padding):
    records = []

    h_step_size = crop_shape[0] - overlap_padding
    w_step_size = crop_shape[1] - overlap_padding
    for i in range(math.ceil(image_shape[0] / h_step_size)):
        for j in range(math.ceil(image_shape[1] / w_step_size)):
            records.append({
                'x': i * h_step_size,
                'y': j * w_step_size,
                'height': crop_shape[0],
                'width': crop_shape[1]
            })
    return records

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def from_numpy(obj):

    if isinstance(obj, dict):
        return {key: from_numpy(value) for key, value in obj.items()}

    if torch.cuda.is_available():
        if isinstance(obj, torch.Tensor): return obj.to('cuda') #float().cuda(non_blocking=True)
        #if isinstance(obj, np.ndarray): return torch.cuda.FloatTensor(obj)
        return obj
    else:
        if isinstance(obj, torch.Tensor): return obj#float()
        #if isinstance(obj, np.ndarray): return torch.FloatTensor(obj)
        return obj

class TransformedDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __getitem__(self, item):
        return self.transform(self.records.iloc[item])

    def __len__(self):
        return len(self.records)

def transform(record):
    image = cv2.imread(record['image_path'])[:, :, [0]]
    crop = image[record['x']: record['x'] + record['height'], record['y']: record['y'] + record['width']]
    pad_bottom = record['height'] - crop.shape[0]
    pad_right = record['width'] - crop.shape[1]

    if (pad_bottom != 0) or (pad_right != 0):
        crop = np.pad(crop, ((0, pad_bottom), (0, pad_right), (0, 0)))

    return {
        'image': np.moveaxis(crop, -1, 0).astype(np.float32) / 255.0,
        **record
    }

def batch_select2d(tensor, indices):
    original_shape = tensor.shape
    indexed_shape = (original_shape[0], indices.shape[1], *original_shape[2:])

    batch_index = (indices >= -1).nonzero()
    batch_index[:, 1] = indices.reshape(-1)
    return tensor[batch_index[:, 0], batch_index[:, 1]].reshape(indexed_shape)

class Prediction:
    def __init__(self, **kwargs):
        self._inner_dict = dict(**kwargs)
        # self._inner_dict['positive_masks'] = self.instance_masks(self._inner_dict['positive_indices'])
        if not self._inner_dict['training']:
            self._inner_dict['top_masks_and_scores'] = self.top_masks_and_scores()

    def __getitem__(self, key):
        return self._inner_dict[key]

    def instance_masks(self, keep_indices, grad_enabled=True):
        with torch.no_grad() if not grad_enabled else contextlib.nullcontext():
            # TODO AS: Don't know how to bake this into model
            num_groups = 4

            mask_encodings, weights = self._inner_dict['mask_encodings'], self._inner_dict['weights']
            output_shape = self._inner_dict['output_shape']

            B, E, H, W = mask_encodings.shape

            if keep_indices is not None:
                weights = batch_select2d(weights.permute(0, 2, 3, 1).reshape(B, -1, (E + 1) * 4), keep_indices)
                weights = weights.reshape(-1, (E + 1), 1, 1)
            else:
                weights = weights.permute(0, 2, 3, 1).reshape(-1, (E + 1), 1, 1)

            masks = torch.nn.functional.conv2d(mask_encodings.reshape(1, E * B, H, W), weights[:, :-1, :, :], groups=B, bias=weights[:, -1, 0, 0])
            masks = torch.nn.functional.sigmoid(masks.view(B, -1, H, W))
            # TODO AS: Does this preserve spatial order?
            masks = [masks[:, i::num_groups] for i in range(num_groups)]
            masks = reduce(operator.mul, masks)
            masks = torch.nn.functional.interpolate(masks, size=output_shape, mode='bilinear')
            return masks

    def top_masks_and_scores(self, top_k=700):
        B = self._inner_dict['objectness'].shape[0]
        indices = self._inner_dict['objectness'].reshape(B, -1).argsort(dim=1, descending=True)[:, :top_k]
        scores = batch_select2d(self._inner_dict['objectness'].reshape(B, -1), indices)
        return self.instance_masks(keep_indices=indices), scores

class CellNet(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        objectness_logits, mask_encodings, weights = self.model(x['image'])

        return Prediction(
            objectness_logits=objectness_logits,
            objectness=objectness_logits.sigmoid(),
            mask_encodings=mask_encodings,
            weights=weights,
            output_shape=x['image'].shape[2:],
            training=self.training,
            # positive_indices=x['positive_indices']
        )

def backform(cropped_mask, x, y, original_shape):
    mask = torch.zeros(original_shape[:2]).to(cropped_mask)
    destination_shape = mask[x: x + cropped_mask.shape[0], y: y + cropped_mask.shape[1]].shape
    if destination_shape != cropped_mask.shape:
        cropped_mask = cropped_mask[:destination_shape[0], :destination_shape[1]]
    mask[x: x + cropped_mask.shape[0], y: y + cropped_mask.shape[1]] = cropped_mask
    return mask

def mask2rle(img):
    '''
    Efficient implementation of mask2rle, from @paulorzp
    --
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    Source: https://www.kaggle.com/xhlulu/efficient-mask2rle
    '''
    pixels = img.flatten()
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# TODO AS: This assumes that masks are sorted score by descending order
def binary_matrix_nms(masks, scores, method='gauss', sigma=0.5):
    N = len(scores)
    masks = masks.reshape(N, -1)
    intersection = torch.mm(masks, masks.T)
    areas = masks.sum(dim=1).expand(N, N)
    union = areas + areas.T - intersection
    ious = (intersection / union).triu(diagonal=1)

    ious_cmax = ious.max(0)[0]
    ious_cmax = ious_cmax.expand(N, N).T
    if method == 'gauss':
        decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)

    decay = decay.min(dim=0)[0]
    return scores * decay

def select_by(mask, tensors):
    return [tensor[mask] for tensor in tensors]

def extract_predictions_from(
    masks, scores,
    min_area, pre_nms_threshold, pre_nms, post_nms_threshold, post_nms_topk,
    matrix_nms_iter, mask_threshold=0.5
):
    # TODO AS: Save memory by doing it in prediction, maybe even all of it?
    # 0. pre nms thresholding
    # masks, scores = select_by(torch.argsort(scores, descending=True)[:pre_nms], [masks, scores])
    masks, scores = select_by(scores > pre_nms_threshold, [masks, scores])

    # 1. maskness
    discrete_masks = (masks > mask_threshold).float()
    maskness = (discrete_masks * masks).sum(dim=(1, 2)) / discrete_masks.sum(dim=(1, 2))
    scores *= maskness
    masks = discrete_masks
    masks, scores = select_by(masks.sum(dim=(1, 2)) > min_area, [masks, scores])

    # 2. pre nms
    masks, scores = select_by(torch.argsort(scores, descending=True)[:pre_nms], [masks, scores])

    if len(masks) < 1:
        return masks, scores

    # 3. nms
    for _ in range(matrix_nms_iter):
        scores = binary_matrix_nms(masks, scores)
        masks, scores = select_by(torch.argsort(scores, descending=True), [masks, scores])

    # 3. a bit of postprocessing
    masks, scores = select_by((scores > post_nms_threshold).nonzero().reshape(-1)[:post_nms_topk], [masks, scores])

    return masks, scores

def predict(
    batch_size=1,
    num_workers=16,
    dataset_path='data/train',
    model_path='data/experiments/20211204_094311_noupscale/best.pth',
    submission_path='submission.csv',
    crop_shape='480x640',
    overlap_padding=32,
    limit=None,
    matrix_nms_iter=4,
    min_area=0,
    pre_nms_threshold=0.1,
    post_nms_threshold=0.15,
    pre_nms=500,
    post_nms_topk=300,
):
    crop_shape = tuple(map(int, crop_shape.split('x')))

    model = CellNet(load_checkpoint(model_path))
    model = as_cuda(model)
    model = model.eval()
    torch.set_grad_enabled(False)

    submission_records = []

    # 1. read each image
    # 2. crop parts of it
    # 3. extract masks with nms
    # 4. backform masks into whole image
    # 5. run nms after all masks were collected
    # 6. calculate metrics

    image_paths = list(glob.glob(f'{dataset_path}/*.png'))
    image_paths = list(map(lambda _id: f'data/train/{_id}.png', [
        '45b966b60d4b', '3bcc8ba1dc17', '1d618b80769f', 'c6a9863504da',
        '1d2396667910', 'a97d5689d4c2', '11c2e4fcac6d', '6064a286cbf3',
        'ae509c50607a', '1874b96fd317', '0c90b86742b2', '100681b6cc7a',
        '41c57fe26957', '6b2f2fab222f', '7d59ab1d21a2', '9bc9775ee371',
        'f2e0ce316b7e', '5863bf795692', '4e99b18bf20f', '3912a0bede5b',
        '499a225c835d', 'b3990528329c', 'bcf94f6bc975', '7ca93f81e669',
        '49d4a04f398c', '4e115eccc68c', '539f24ebc61d', 'a55a105360b8',
        '4698edfd5878', 'ac877991fa24', '446cf8ba65e5', '4d52c84bfe79',
        'f00798e9b1eb', 'f0e54d645fe5', '042dc0e561a4', 'bc0b9c1ff4dc',
        '07e9ba109e34', '15283b194621', '87aabe7ab3a5', 'd96878ba3ab6',
        '815de003cb5b', '44c353126f35', '98fd9ed43654', '9f1c2cfc936f',
        'cc40345857dd', '8a754409504b', 'ad30ecfc1682'
    ]))
    for image_path in image_paths[:limit]:
        image = cv2.imread(image_path)
        crop_records = generate_crop_parameters(image.shape, crop_shape=crop_shape, overlap_padding=overlap_padding)
        df = pd.DataFrame(crop_records)
        df['image_path'] = image_path
        dataset = TransformedDataset(df, partial(transform))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        masks_and_scores = []

        for batch in tqdm.tqdm(dataloader):
            batch = from_numpy(batch)
            outputs = model(batch)

            for i in range(len(batch['image'])):
                pred_masks, scores = extract_predictions_from(
                    *outputs.top_masks_and_scores(),
                    min_area=min_area, pre_nms_threshold=pre_nms_threshold, pre_nms=pre_nms,
                    post_nms_threshold=post_nms_threshold, post_nms_topk=post_nms_topk,
                    matrix_nms_iter=matrix_nms_iter
                )
                if len(pred_masks) > 0:
                    pred_masks = torch.stack([backform(mask, batch['x'][i], batch['y'][i], image.shape) for mask in pred_masks])
                    masks_and_scores.append((
                        pred_masks,
                        scores
                    ))

        masks, scores = zip(*masks_and_scores)
        masks, scores = torch.cat(masks), torch.cat(scores)
        masks, scores = extract_predictions_from(masks, scores,
            min_area=min_area, pre_nms_threshold=pre_nms_threshold, pre_nms=pre_nms,
            post_nms_threshold=post_nms_threshold, post_nms_topk=post_nms_topk,
            matrix_nms_iter=matrix_nms_iter
        )

        for mask in masks:
            rle = mask2rle(mask.cpu().numpy())
            submission_records.append({
                'id': os.path.splitext(os.path.basename(image_path))[0],
                'predicted': rle
            })

    pd.DataFrame(submission_records).to_csv(submission_path, index=False)


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