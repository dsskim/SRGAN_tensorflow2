import os
import argparse
import tensorflow as tf
from model import Generator, Discriminator

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', type=int, help='epoch to train the network', default=10)
    parser.add_argument('--lr_image_size', help='the image size', default=28)
    parser.add_argument('--hr_image_size', help='the image size', default=28)
    parser.add_argument('--initial_learning_rate', type=float, help='Initial learning rate', default=1e-1)
    parser.add_argument('--train_batch_size', type=int, help='batch size to train network', default=128)

    args = parser.parse_args()
    return args

def main(args):
    gene = Generator(args.lr_image_size)
    disc = Discriminator(args.hr_image_size)

if __name__ == '__main__':
    # Setting GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    main(get_parser())