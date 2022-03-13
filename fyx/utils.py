import torch
import numpy as np

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

def to_numpy(obj):
    if isinstance(obj, dict):
        return {key: to_numpy(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(map(to_numpy, obj))
    return obj.data.cpu().numpy()

def encode_rle(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle(shape, encoded_mask):
    if encoded_mask == 'nan': return np.zeros(shape)
    shape = shape[::-1]
    numbers = np.array(list(map(int, encoded_mask.split())))
    starts, lengths = numbers[::2], numbers[1::2]
    # Enumerates from 1
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends): mask[start:end] += 1
    mask = np.clip(mask, a_min=0, a_max=2)
    return mask.reshape(shape).T