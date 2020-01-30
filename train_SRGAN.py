import os
import argparse
import time
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from model import Generator, Discriminator, Content_Net
from dataset import DIV2K


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_train_step', type=int, help='epoch to train the network', default=200000)
    parser.add_argument('--batch_size', type=int, help='batch size to train the network', default=16)
    parser.add_argument('--learning_rate', help='learning rate to train the network', default=[1e-4, 1e-5])
    parser.add_argument('--learning_schedule', help='learning rate to train the network', default=[100000])
    parser.add_argument('--train_lr_size', type=int, help='the image size', default=24)
    parser.add_argument('--train_scale', type=int, help='the super resolution scale', default=4)
    parser.add_argument('--output_folder', help='the output folder to train', default='./output')
    parser.add_argument('--pre_trained_weight_path', help='the weight path for train', default='./output/pre_weights')
    parser.add_argument('--valid_image_path', help='the image path to validation', default='./test_img/0851x4.png')
    parser.add_argument('--interval_save_weight', type=int, help='interval to save ckpt', default=1000)
    parser.add_argument('--interval_validation', type=int, help='interval to test validation', default=1000)
    parser.add_argument('--interval_show_info', type=int, help='interval to show information', default=100)

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

    hr_feature = content_model(hr) / 12.75
    sr_feature = content_model(sr) / 12.75

    return mse(hr_feature, sr_feature)


def main(args):

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    model_subs_path = os.path.join(args.output_folder, 'arch_img')
    if not os.path.exists(model_subs_path):
        os.mkdir(model_subs_path)
    weights_path = os.path.join(args.output_folder, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    valid_img_save_path = os.path.join(args.output_folder, 'valid_img')
    if not os.path.exists(valid_img_save_path):
            os.mkdir(valid_img_save_path)

    # Dataset
    train_loader = DIV2K(scale=args.train_scale, downgrade='unknown', subset='train')

    # Create a tf.data.Dataset
    train_ds = train_loader.dataset(batch_size=args.batch_size, random_transform=True)

    gene = Generator(args.train_lr_size)

    ## load pre_trained_weights
    files_path = os.path.join(args.pre_trained_weight_path, '*.h5')
    latest_file_path = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)[0]
    gene.load_weights(latest_file_path)

    disc = Discriminator(args.train_lr_size * args.train_scale)
    tf.keras.utils.plot_model(disc, to_file=os.path.join(model_subs_path, 'discriminator.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
    content_model = Content_Net(args.train_lr_size * args.train_scale)
    tf.keras.utils.plot_model(content_model, to_file=os.path.join(model_subs_path, 'content.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
    
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(args.learning_schedule, args.learning_rate)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_fn)

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

    def valid_step(image_path, weights_path, step):
        img = np.array(Image.open(image_path))[..., :3]

        scaled_lr_img = tf.cast(img, tf.float32)
        scaled_lr_img = scaled_lr_img / 255
        scaled_lr_img = scaled_lr_img[np.newaxis,:,:,:]

        gene = Generator(None)
        
        files_path = os.path.join(weights_path, 'gen_step_*')
        latest_file_path = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)[0]
        gene.load_weights(latest_file_path)
        
        sr_img = gene(scaled_lr_img).numpy()
        
        sr_img = np.clip(sr_img, -1, 1)
        sr_img = (sr_img + 1) * 127.5
        sr_img = np.around(sr_img)
        sr_img = sr_img.astype(np.uint8)
        
        im = Image.fromarray(sr_img[0])

        im.save(os.path.join(valid_img_save_path, 'step_{0:07d}.png'.format(step)))
        print('[step_{}.png] save images'.format(step))

    start = time.time()
    for step, (lr, hr) in enumerate(train_ds.take(args.max_train_step + 1)):
        ## re-scale
        ## lr: 0 ~ 1
        ## hr: -1 ~ 1
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        lr = lr / 255
        hr = hr / 127.5 - 1

        gen_loss, disc_loss = train_step(lr, hr, gene, disc, content_model)

        if step % args.interval_save_weight == 0:
            gene.save_weights(os.path.join(weights_path, 'gen_step_{}.h5'.format(step)))
            disc.save_weights(os.path.join(weights_path, 'disc_step_{}.h5'.format(step)))

        if step % args.interval_validation == 0:
            valid_step(args.valid_image_path, weights_path, step)

        if step % args.interval_show_info == 0:
            print('step:{}/{} gen_loss {}, disc_loss {}, Training time is {} step/s'.format(step, args.max_train_step, gen_loss, disc_loss, (time.time() - start) / args.interval_show_info))
            start = time.time()


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
