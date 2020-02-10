from .layers import conv, deconv
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np

DISP_SCALING = 10
MIN_DISP = 0.01


def disp_branch(image, l2=0.05):
    """Predict inverse of depth from a single image."""
    h = image.get_shape()[1]
    w = image.get_shape()[2]

    # features, kernel size, stride
    encoder = [
        (32, 7, 2),
        (32, 7, 1),
        (64, 5, 2),
        (64, 5, 1),
        (128, 3, 2),
        (128, 3, 1),
        (256, 3, 2),
        (256, 3, 1),
        (512, 3, 2),
        (512, 3, 1),
        (512, 3, 2),
        (512, 3, 1),
        (512, 3, 2),
        (512, 3, 1),
    ]
    y = image
    encoder_layers = []
    # build encoder
    for features, kz, stride in encoder:
        print("encoder:", features, kz, stride, y.shape)
        y = conv(y, features, kz, stride=stride, l2=l2)
        encoder_layers.append(y)

    # decoder

    # simple concat with encoder layers
    decoder_concat = [
        (512, 3),
        (512, 3),
        (256, 3),
    ]
    for i, (features, kz) in enumerate(decoder_concat):
        print("decoder(concat): ", features, kz, y.shape)
        # There might be dimension mismatch due to uneven down/up-sampling.
        y = deconv(y, features, kz, stride=2, l2=l2)
        encoder_layer = encoder_layers[-1 + (-2) * (i + 1)]
        y = _resize_like(y, encoder_layer)
        y = tf.concat([y, encoder_layer], axis=3)
        y = conv(y, features, kz, stride=1, l2=l2)

    y = deconv(y, 128, 3, stride=2, l2=l2)
    y = tf.concat([y, encoder_layers[-1 + (-2) * 4]], axis=3)
    disp4 = conv(y, 1, 3, stride=1, activation='sigmoid', norm=False, l2=l2) * DISP_SCALING + MIN_DISP
    disp4_up = tf.image.resize(disp4, (np.int(h / 4), np.int(w / 4)), tf.image.ResizeMethod.BILINEAR)

    y = deconv(y, 64, 3, stride=2, l2=l2)
    y = tf.concat([y, encoder_layers[-1 + (-2) * 5], disp4_up], axis=3)
    disp3 = conv(y, 1, 3, stride=1, activation='sigmoid', norm=False, l2=l2) * DISP_SCALING + MIN_DISP
    disp3_up = tf.image.resize(disp3, (np.int(h / 2), np.int(w / 2)), tf.image.ResizeMethod.BILINEAR)

    y = deconv(y, 32, 3, stride=2, l2=l2)
    y = tf.concat([y, encoder_layers[-1 + (-2) * 6], disp3_up], axis=3)
    disp2 = conv(y, 1, 3, stride=1, activation='sigmoid', norm=False, l2=l2) * DISP_SCALING + MIN_DISP
    disp2_up = tf.image.resize(disp2, (h, w), tf.image.ResizeMethod.BILINEAR)

    y = deconv(y, 16, 3, stride=2, l2=l2)
    y = tf.concat([y, disp2_up], axis=3)
    disp1 = conv(y, 1, 3, stride=1, activation='sigmoid', norm=False, l2=l2) * DISP_SCALING + MIN_DISP

    return disp1, disp2, disp3, disp4


def disp_model(image_stack, i, seq_length, l2=0.05):
    image = image_stack[:, :, :, 3 * i:3 * (i + 1)]
    disps = disp_branch(image, l2=l2)
    return Model(inputs=image_stack, outputs=disps)


def _resize_like(inputs, ref):
    i_h, i_w = inputs.get_shape()[1], inputs.get_shape()[2]
    r_h, r_w = ref.get_shape()[1], ref.get_shape()[2]
    if i_h == r_h and i_w == r_w:
        return inputs
    else:
        return tf.image.resize(inputs, [r_h, r_w], tf.image.ResizeMethod.NEAREST_NEIGHBOR)


if __name__ == '__main__':
    from tensorflow.keras import Input

    i = Input((128, 416, 3))
    model = disp_model(i)
    print(model.outputs)
