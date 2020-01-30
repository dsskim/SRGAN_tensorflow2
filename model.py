import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Dense,
    BatchNormalization,
    PReLU,
    LeakyReLU
)
from tensorflow.keras.applications import (
    VGG19
)


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def res_block(x):
    skip_layer = x

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=_regularizer)(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=_regularizer)(x)
    x = BatchNormalization()(x)

    out = tf.add(x, skip_layer)

    return out


def Generator(size=None, channels=3, res_block_repeat=16, scale=4, name='generator'):
    inputs = Input([size, size, channels], name='inputs')

    x = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', use_bias=False, kernel_regularizer=_regularizer)(inputs)
    x = PReLU(shared_axes=[1, 2])(x)

    skip_layer = x

    for _ in range(res_block_repeat):
        x = res_block(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=_regularizer)(x)
    x = BatchNormalization()(x)
    x = tf.add(x, skip_layer)

    for _ in range(int(math.sqrt(scale))):
        x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=_regularizer)(x)
        x = PReLU(shared_axes=[1, 2])(x)

    out = Conv2D(filters=3, kernel_size=9, strides=1, padding='same', use_bias=False, kernel_regularizer=_regularizer, activation='tanh')(x)

    return Model(inputs, out, name=name)


def Discriminator(size=None, channels=3, name='discriminator'):
    inputs = Input([size, size, channels], name='inputs')

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=_regularizer)(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=_regularizer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    depth = 128
    for _ in range(3):
        x = Conv2D(filters=depth, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=_regularizer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=depth, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=_regularizer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        depth *= 2

    x = Flatten()(x)
    x = Dense(1024, use_bias=False, kernel_regularizer=_regularizer)(x)
    x = LeakyReLU(alpha=0.2)(x)

    out = Dense(1, use_bias=False, kernel_regularizer=_regularizer, activation='sigmoid')(x)

    return Model(inputs, out, name=name)


def Content_Net(size=None, channels=3, i=5, j=4):
    vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(size, size, channels))
    block_name = 'block{}_conv{}'.format(i, j)

    vgg19.trainable = False
    # Make trainable as False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer(block_name).output)
    model.trainable = False

    return model
