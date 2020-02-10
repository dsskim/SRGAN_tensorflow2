import os
import argparse
import time
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from model import Generator
from loss import mse_based_loss
from dataset import DIV2K


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_pre_train_step', type=int, help='epoch to train the network', default=1000000)
    parser.add_argument('--batch_size', type=int, help='batch size to train the network', default=16)
    parser.add_argument('--train_lr_size', type=int, help='the image size', default=24)
    parser.add_argument('--train_scale', type=int, help='the super resolution scale', default=4)
    parser.add_argument('--learning_rate', help='learning rate to train the network', default=1e-4)
    parser.add_argument('--output_folder', help='the output folder to train', default='./output')
    parser.add_argument('--valid_image_path', help='the image path to validation', default='./test_img/0851x4.png')
    parser.add_argument('--interval_save_weight', type=int, help='interval to save ckpt', default=1000)
    parser.add_argument('--interval_validation', type=int, help='interval to test validation', default=1000)
    parser.add_argument('--interval_show_info', type=int, help='interval to show information', default=100)

    args = parser.parse_args()
    return args


def main(args):

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    model_subs_path = os.path.join(args.output_folder, 'arch_img')
    if not os.path.exists(model_subs_path):
        os.mkdir(model_subs_path)
    weights_path = os.path.join(args.output_folder, 'pre_weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    valid_img_save_path = os.path.join(args.output_folder, 'pre_valid_img')
    if not os.path.exists(valid_img_save_path):
        os.mkdir(valid_img_save_path)

    # Dataset
    train_loader = DIV2K(scale=args.train_scale, downgrade='unknown', subset='train') # 'bicubic', 'unknown', 'mild' or 'difficult'

    # Create a tf.data.Dataset
    train_ds = train_loader.dataset(batch_size=args.batch_size, random_transform=True)

    # Define Model
    gene = Generator(args.train_lr_size, scale=args.train_scale)
    tf.keras.utils.plot_model(gene, to_file=os.path.join(model_subs_path, 'generator.png'), show_shapes=True, show_layer_names=True, expand_nested=True)
    generator_optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    @tf.function
    def pre_train_step(lr, hr, generator):
        with tf.GradientTape() as gen_tape:
            sr = generator(lr, training=True)
            loss = mse_based_loss(sr, hr)

        gradients_of_generator = gen_tape.gradient(loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

        return loss

    def valid_step(image_path, weights_path, step):
        img = np.array(Image.open(image_path))[..., :3]

        scaled_lr_img = tf.cast(img, tf.float32)
        scaled_lr_img = scaled_lr_img / 255
        scaled_lr_img = scaled_lr_img[np.newaxis,:,:,:]

        gene = Generator(None)
        
        files_path = os.path.join(weights_path, '*.h5')
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

    # Start Train
    start = time.time()
    for pre_step, (lr, hr) in enumerate(train_ds.take(args.max_pre_train_step + 1)):
        ## re-scale
        ## lr: 0 ~ 1
        ## hr: -1 ~ 1
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        lr = lr / 255
        hr = hr / 127.5 - 1

        mse = pre_train_step(lr, hr, gene)

        if pre_step % args.interval_save_weight == 0:
            gene.save_weights(os.path.join(weights_path, 'pre_gen_step_{}.h5'.format(pre_step)))

        if pre_step % args.interval_validation == 0:
            valid_step(args.valid_image_path, weights_path, pre_step)

        if pre_step % args.interval_show_info == 0:
            print('step:{}/{} MSE_LOSS {}, Training time is {} step/s'.format(pre_step, args.max_pre_train_step, mse, (time.time() - start) / args.interval_show_info))
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
