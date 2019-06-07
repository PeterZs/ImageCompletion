from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class ImgDataset(Dataset):
    """Image dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.img_paths = self.__get_all_img_paths(self.root_dir)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx, color_format='RGB'):
        img = Image.open(self.img_paths[idx])
        img = img.convert(color_format)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __get_all_img_paths(self, root_dir):
        img_paths = []
        root_dir = os.path.expanduser(root_dir)
        for path in os.listdir(root_dir):
            path = os.path.join(root_dir, path)
            img_paths.append(path)
        return img_paths
