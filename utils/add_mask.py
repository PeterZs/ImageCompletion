import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from .generate_random_holes import gen_input_mask, gen_hole_area


def add_mask(args, device, batch_data, hole_area, mpv):
    mask = gen_input_mask(
        shape=(batch_data.shape[0], 1, batch_data.shape[2], batch_data.shape[3]),
        hole_size=((args.hole_min_w, args.hole_max_w), (args.hole_min_h, args.hole_max_h)),
        hole_area=hole_area,
        max_holes=args.max_holes,
    ).to(device)
    batch_data_with_mask = batch_data - batch_data * mask + mpv * mask
    input_batch = torch.cat((batch_data_with_mask, mask), dim=1)
    return mask, input_batch, batch_data_with_mask


def get_train_mean(args, dataset, device):
    # compute mean pixel value of training dataset
    mpv = np.zeros(shape=(3,))
    if args.mpv is None:
        pbar = tqdm(total=len(dataset.img_paths), desc='computing mean pixel value for training dataset...')
        for imgpath in dataset.img_paths:
            img = Image.open(imgpath)
            x = np.array(img, dtype=np.float32) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(dataset.img_paths)
        pbar.close()
    else:
        mpv = np.array(args.mpv)

    # save training config
    mpv_json = []
    for i in range(3):
        mpv_json.append(float(mpv[i]))  # convert to json serializable type
    args_dict = vars(args)
    args_dict['mpv'] = mpv_json
    with open('config.json', mode='w') as f:
        json.dump(args_dict, f)

    # make mpv & alpha tensor
    mpv = torch.tensor(mpv.astype(np.float32).reshape(1, 3, 1, 1)).to(device)

    return mpv
