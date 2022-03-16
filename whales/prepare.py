
import fire
import pandas as pd
import tqdm
import numpy as np
import json
import pycocotools.mask

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

def prepare(
    input_path,
    output_path,
    **kwargs
):
    assert len(kwargs) == 0, f'Unrecognized args: {kwargs}'

    with open(input_path) as f:
        dataset = json.load(f)

    output_records = []
    images_df = pd.DataFrame(dataset['images'])
    annotations_df = pd.DataFrame(dataset['annotations'].values())

    for _, image_record in tqdm.tqdm(images_df.iterrows(), total=len(images_df)):
        image_annotations = annotations_df[annotations_df['image_id'] == image_record['id']]
        height, width = image_record['height'], image_record['width']
        default_annotation_fields = {
            'id': image_record['file_name'].split('.')[0],
            'width': width,
            'height': height,
            'cell_type': image_record['file_name'].split('_')[0],
            'plate_time': None,
            'sample_date': None,
            'sample_id': None,
            'elapsed_timedelta': None
        }

        # TODO AS: Currently merging all masks into one
        masks = pycocotools.mask.decode(pycocotools.mask.frPyObjects(image_annotations['segmentation'].sum(), height, width))
        combined_mask = np.logical_or.reduce(masks, axis=-1).astype(np.uint8)[:, :, None]
        rle = mask2rle(combined_mask)
        annotation = {
            **default_annotation_fields,
            'annotation': rle
        }
        output_records.append(annotation)

    pd.DataFrame(output_records).to_csv(output_path, index=False)

if __name__ == '__main__':
    fire.Fire(prepare)