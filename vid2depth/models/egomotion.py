from tensorflow.keras.models import Model
import tensorflow as tf
from .layers import conv

EGOMOTION_VEC_SIZE = 6


def egomotion_model(image_stack, scaling=0.01, l2=0.05):
    """
    Args:
    image_stack: Input tensor with shape [B, h, w, seq_length * 3]

    """
    seq_length = image_stack.shape[3]
    num_egomotion_vecs = seq_length - 1

    arch = [
        (16, 7),
        (32, 5),
        (64, 3),
        (128, 3),
        (256, 3),
        (256, 3),
        (256, 3),
    ]

    y = image_stack
    for features, kz in arch:
        y = conv(y, features, kz, stride=2, l2=l2)

    #
    pred_channels = EGOMOTION_VEC_SIZE * num_egomotion_vecs
    ego_pred = conv(y, pred_channels, 1, stride=1, norm=None, activation=None, l2=l2)
    ego_avg = tf.reduce_mean(ego_pred, [1, 2])

    # Tinghui found that scaling by a small constant facilitates training.
    ego_final = scaling * tf.reshape(ego_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])
    return Model(image_stack, ego_final)


if __name__ == '__main__':
    from tensorflow.keras import Input
    inputs = Input((128, 416, 9))
    model = egomotion_model(inputs)
    print(model.outputs)
