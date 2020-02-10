import tensorflow as tf


def discriminator_loss(fake_output, real_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def content_loss(content_model, sr, hr):
    # real_output, fake_output range: -1 ~ 1
    mse = tf.keras.losses.MeanSquaredError()

    hr_feature = content_model(hr) / 12.75
    sr_feature = content_model(sr) / 12.75

    return mse(hr_feature, sr_feature)


def mse_based_loss(sr, hr):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(sr, hr)
