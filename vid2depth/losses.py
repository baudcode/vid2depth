import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def depth_smoothness(depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))


def ssim(x, y):
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = AveragePooling2D(3, 1, 'VALID')(x)
    mu_y = AveragePooling2D(3, 1, 'VALID')(y)
    sigma_x = AveragePooling2D(3, 1, 'VALID')(x**2) - mu_x**2
    sigma_y = AveragePooling2D(3, 1, 'VALID')(y**2) - mu_y**2
    sigma_xy = AveragePooling2D(3, 1, 'VALID')(x * y) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)
