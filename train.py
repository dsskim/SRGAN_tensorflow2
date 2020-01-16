import os
import argparse
import time
import tensorflow as tf
from model import Generator, Discriminator, Content_Net


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', type=int, help='epoch to train the network', default=10)
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
    hr_feature = content_model.predict(hr)
    sr_feature = content_model.predict(sr)

    return mse(hr_feature, sr_feature)


def main(args):
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
    #
    # for epoch in range(args.max_epoch):
    #     start = time.time()
    #
    #     for image_batch in dataset:
    #         gen_loss, disc_loss = train_step(image_batch, gene, disc, content_model)
    #
    #     # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
    #     print('gen_loss {}, disc_loss {}, Time for epoch {} is {} sec'.format(gen_loss, disc_loss, epoch + 1,
    #                                                                           time.time() - start))

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
