from vid2depth.losses import depth_smoothness_loss, ssim_loss, loss_fn
import numpy as np
import tensorflow as tf
import cv2


def test_ssim():
    image = np.ones((1, 32, 32, 3), np.float32)
    ssim_value = tf.reduce_mean(ssim_loss(image, image))
    assert(ssim_value.numpy() == 0.0)

    ds = depth_smoothness_loss(image, image, 0)
    assert(ds.numpy() == 0.0)
