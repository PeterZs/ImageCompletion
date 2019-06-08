import argparse

parser = argparse.ArgumentParser(description='Pytorch train operation')

parser.add_argument('--data_path',
                    default="datasets/celebA",
                    help='path of train and test dataset')

parser.add_argument('--result_path',
                    default="weights/celebA",
                    help='path of train and test dataset')

parser.add_argument('--weight_completion',
                    default=None,
                    help='load completion weights ')

parser.add_argument('--weight_discriminator',
                    default=None,
                    help='load discriminator weights ')

parser.add_argument('--iteration1',
                    default=0,
                    type=int,
                    help='number of iteration of phase 1 to train')

parser.add_argument('--iteration2',
                    default=200,
                    type=int,
                    help='number of iteration of phase 1 to train')

parser.add_argument('--iteration3',
                    default=10000,
                    type=int,
                    help='number of iteration of phase 1 to train')

parser.add_argument('--test_period1',
                    default=100,
                    type=int,
                    help='number of iteration of phase 1 to test and save snapshot')

parser.add_argument('--test_period2',
                    default=100,
                    type=int,
                    help='number of iteration of phase 2 to test and save snapshot')

parser.add_argument('--test_period3',
                    default=1000,
                    type=int,
                    help='number of iteration of phase 3 to test and save snapshot')

parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help='the batch size for training')

parser.add_argument("--test_batch_size",
                    type=int,
                    default=16,
                    help='the batch size for testing')

parser.add_argument('--dataset',
                    default='celeba',
                    help='initial learning_rate')

parser.add_argument('--mpv',
                    nargs=3,
                    type=float,
                    default=None,
                    help='mean pixel value')

parser.add_argument('--alpha',
                    type=float,
                    default=4e-4,
                    help='alpha value for gan loss')

parser.add_argument('--lr',
                    default=0.01,
                    help='initial learning_rate')


parser.add_argument('--min_hole_width',
                    type=int,
                    default=48)

parser.add_argument('--max_hole_width',
                    type=int,
                    default=96)

parser.add_argument('--min_hole_height',
                    type=int,
                    default=48)

parser.add_argument('--max_hole_height',
                    type=int,
                    default=96)

parser.add_argument('--full_image_size',
                    type=int,
                    default=160)

parser.add_argument('--local_patch_size',
                    type=int,
                    default=96)

opt = parser.parse_args()
