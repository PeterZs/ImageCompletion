import logging
import os
import argparse
import random
import shutil


def split_data(args):
    # check if path exits
    if not os.path.exists(args.src_data_dir):
        logging.warning('path not exits!')
        return

    logging.info('Begin split dataset')

    img_path = []
    for file in os.listdir(args.src_data_dir):
        path = os.path.join(args.src_data_dir, file)
        img_path.append(path)
    random.shuffle(img_path)

    # separate the paths
    border = int(args.proportion * len(img_path))
    train_paths = img_path[:border]
    test_paths = img_path[border:]
    logging.info('train set size: %d images.' % len(train_paths))
    logging.info('test set size: %d images.' % len(test_paths))

    # create dst directories
    train_dir = os.path.join(args.dst_data_dir, 'train')
    test_dir = os.path.join(args.dst_data_dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    logging.info('dataset is making, please wait.')
    for src_img_train_path in train_paths:
        dst_path = os.path.join(train_dir, os.path.basename(src_img_train_path))
        shutil.move(src_img_train_path, dst_path)
    for src_img_test_path in test_paths:
        dst_path = os.path.join(test_dir, os.path.basename(src_img_test_path))
        shutil.move(src_img_test_path, dst_path)
    logging.info('work done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset split setting')
    parser.add_argument('src_data_dir', type=str, help='path to the source whole data set')
    parser.add_argument('--dst_data_dir', type=str, default='datasets', help='path to the dst dataset')
    parser.add_argument('--proportion', type=float, default=0.8, help='proportion of training data(default 0.8)')

    args = parser.parse_args()
    args.src_data_dir = os.path.expanduser(args.src_data_dir)
    split_data(args)
