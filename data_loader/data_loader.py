import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from .img_dataset import ImgDataset
import random


def train_data_loader(args):
    data_transform = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])

    train_data_set = ImgDataset(os.path.join(args.data_path, 'train'), data_transform)
    # test_data_set = ImgDataset(os.path.join(args.data_dir, 'test'), data_transform)
    train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)

    return train_data_set, train_loader


def test_data_loader(args):
    data_transform = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])

    # train_data_set = ImgDataset(os.path.join(args.data_dir, 'train'), data_transform)
    test_data_set = ImgDataset(os.path.join(args.data_path, 'test'), data_transform)
    test_loader = DataLoader(test_data_set, batch_size=args.test_batch_size, shuffle=True)

    return test_data_set, test_loader


def sample_random_batch(dataset, batch_size=64):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)
