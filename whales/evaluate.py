import os
import cv2
import skimage
import fire
import numpy as np
import pandas as pd

from sartorius.fit import rle2mask, merge_into_single_image, image_mean_average_precision


def evaluate(pred_csv='submission.csv', gt_csv='data/train.csv'):
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)

    os.makedirs('data/evaluation', exist_ok=True)

    values = []
    for _id in pred_df['id'].unique():
        key = 'annotation' if 'annotation' in pred_df else 'predicted'
        pred_masks = pred_df[pred_df['id'] == _id][key].values
        pred_masks = np.stack([rle2mask(pred_mask, (520, 704)) for pred_mask in pred_masks])

        key = 'annotation' if 'annotation' in gt_df else 'predicted'
        gt_masks = gt_df[gt_df['id'] == _id][key].values
        gt_masks = np.stack([rle2mask(gt_mask, (520, 704)) for gt_mask in gt_masks])

        image = cv2.imread(f'data/train/{_id}.png')

        pred_img = skimage.color.label2rgb(merge_into_single_image(pred_masks, np.ones(len(pred_masks)), image.shape[:2]), image=image, bg_label=0)
        gt_img = skimage.color.label2rgb(merge_into_single_image(gt_masks, np.ones(len(gt_masks)), image.shape[:2]), image=image, bg_label=0)
        cv2.imwrite(f'data/evaluation/{_id}.png', np.vstack([pred_img, gt_img]) * 255)

        values.append(image_mean_average_precision(pred_masks, gt_masks))

    print(np.mean(values))


if __name__ == '__main__':
    fire.Fire(evaluate)