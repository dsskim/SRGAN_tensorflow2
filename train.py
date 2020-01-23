import os
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import Generator, Discriminator, Content_Net
from dataset import DIV2K


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', type=int, help='epoch to train the network', default=100)
    parser.add_argument('--lr_image_size', help='the image size', default=24)
    parser.add_argument('--hr_image_size', help='the image size', default=96)

    args = parser.parse_args()
    return args


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def content_loss(content_model, hr, sr):
    # real_output, fake_output range: -1 ~ 1

    mse = tf.keras.losses.MeanSquaredError()
    hr_feature = content_model(hr)
    sr_feature = content_model(sr)

    return mse(hr_feature, sr_feature)


def generate_and_save_images(model, epoch, input):
    import matplotlib.pyplot as plt

    predictions = model(input, training=False)

    fig, axs = plt.subplots(4, 4)
    for i, ax in enumerate(axs.flat):
        ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def main(args):

    # Dataset
    train_loader = DIV2K(scale=4,  # 2, 3, 4 or 8
                         downgrade='unknown',  # 'bicubic', 'unknown', 'mild' or 'difficult'
                         subset='train')  # Training dataset are images 001 - 800
    valid_loader = DIV2K(scale=4,  # 2, 3, 4 or 8
                         downgrade='unknown',  # 'bicubic', 'unknown', 'mild' or 'difficult'
                         subset='valid')  # Validation dataset are images 801 - 900

    # Create a tf.data.Dataset
    train_ds = train_loader.dataset(batch_size=16,
                                    random_transform=True,
                                    repeat_count=1)
    # Create a tf.data.Dataset
    valid_ds = valid_loader.dataset(batch_size=1,  # use batch size of 1 as DIV2K images have different size
                                    random_transform=False,  # use DIV2K images in original size
                                    repeat_count=1)  # 1 epoch

    gene = Generator(args.lr_image_size)
    tf.keras.utils.plot_model(gene, to_file='generator.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    disc = Discriminator(args.hr_image_size)
    tf.keras.utils.plot_model(disc, to_file='discriminator.png', show_shapes=True, show_layer_names=True, expand_nested=True)
    content_model = Content_Net(args.hr_image_size)
    tf.keras.utils.plot_model(content_model, to_file='content.png', show_shapes=True, show_layer_names=True, expand_nested=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(lr, hr, generator, discriminator, content):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            sr = generator(lr, training=True)

            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)

            gen_loss = generator_loss(sr_output)
            cont_loss = content_loss(content, hr, sr)
            perceptual_loss = cont_loss + 1e-3 * gen_loss

            disc_loss = discriminator_loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perceptual_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return perceptual_loss, disc_loss

    for epoch in range(args.max_epoch):
        start = time.time()

        for lr, hr in train_ds:
            # pre-processing
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            lr = lr / 127.5 - 1
            hr = hr / 127.5 - 1

            gen_loss, disc_loss = train_step(lr, hr, gene, disc, content_model)

        print('epoch:{}/{} gen_loss {}, disc_loss {}, Time for epoch {} is {} sec'.format(epoch,
                                                                                          args.max_epoch,
                                                                                          gen_loss,
                                                                                          disc_loss,
                                                                                          epoch + 1,
                                                                                          time.time() - start))

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
