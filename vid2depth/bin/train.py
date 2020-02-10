from ..models import vid2depth
from ..losses import loss_fn
from ..summaries import build_summaries
from ..settings import logger
from ..datasets.image import ImageDataset, DataType
from ..datasets.utils import convert2tfdataset
from ..augment import preprocess, get_augment_fn

import tensorflow as tf
import tqdm
import os
import argparse

import multiprocessing


def simple_test_model(ds, model):

    def loss(model, x, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        output = model(x, training=training)
        image_stack, intrinsic_mat, intrinsic_mat_inv = x
        print(image_stack.shape, intrinsic_mat.shape, intrinsic_mat_inv.shape)

        print('ego:', output['egomotion'].shape)
        for seq_id, disp in output['disparities'].items():
            print("disp (seq=%d):" % seq_id, [d.shape for d in disp])

        for seq_id, depth in output['depths'].items():
            print("depth (seq=%d):" % seq_id, [d.shape for d in depth])

        return loss_fn(x, output)

    for x in ds:
        break

    total_loss, _, _ = loss(model, x, training=False)
    print("Loss test: {}".format(total_loss))


def train(ds, model, steps_per_epoch=1000, logdir='./logs', epochs=10, lr=0.0002, beta1=0.9, summary_freq=100,
          reconstr_weight=0.85, ssim_weight=0.15, smooth_weight=0.05,
          batch_size=1, height=128, width=416, seq_length=3, num_scales=4, l2=0.05):

    def grad(model, x):
        with tf.GradientTape() as tape:
            output = model(x, training=True)
            loss_value, losses, loss_data = loss_fn(x, output,
                                                    reconstr_weight=reconstr_weight, ssim_weight=ssim_weight, smooth_weight=smooth_weight,
                                                    batch_size=batch_size, height=height, width=width, seq_length=seq_length, l2=l2,
                                                    num_scales=num_scales)

        gradient = tape.gradient(loss_value, model.trainable_variables)
        return loss_value, losses, loss_data, output, gradient

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
    train_loss_results = []

    writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    step = 0

    with writer.as_default():
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            with tqdm.tqdm(total=steps_per_epoch, unit='step', desc='training epoch %d' % epoch) as tq:
                for x in ds:
                    # Optimize the model
                    loss_value, losses, loss_data, output, grads = grad(model, x)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    epoch_loss_avg(loss_value)

                    if step % summary_freq == 0:
                        build_summaries(output, losses, loss_data, seq_length, num_scales, step)

                    tq.set_postfix(**{l: v.numpy() for l, v in losses.items()})

                    tq.update(1)
                    step += 1

                    if step % steps_per_epoch == 0:
                        break

                train_loss_results.append(epoch_loss_avg.result())

                if epoch % 1 == 0:
                    print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

    print(train_loss_results)


def get_args(args=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # training parameters
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('-l', '--logdir', default='./logs', help='log directory')
    parser.add_argument('-sfreq', '--summary_freq', type=int, default=100, help='summary frequency steps')

    # model specific parameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('-b1', '--beta1', type=float, default=0.9, help='beta1 for adam optimizer')
    parser.add_argument('--reconstr_weight', type=float, default=0.85, help='reconstruction loss weight')
    parser.add_argument('--ssim_weight', type=float, default=0.15, help='ssim loss weight')
    parser.add_argument('--smooth_weight', type=float, default=0.05, help='smooth loss weight')
    parser.add_argument('-l2', '--weight_norm', type=float, default=0.05, help='weight normalization')

    # data specific parameters
    parser.add_argument('--height', type=int, default=128, help='image height')
    parser.add_argument('--width', type=int, default=416, help='image width')
    parser.add_argument('-sl', '--seq_length', type=int, default=3, help='sequence length')
    parser.add_argument('-se', '--sample_every', type=int, default=6, help='sample every x frames')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='data directory')
    parser.add_argument('-shubufsize', '--shuffle_buffer_size', type=int, default=50, help='shuffle buffer size')

    return parser.parse_args(args=args)


def train_model(args):

    NUM_SCALES = 4

    # create the dataset
    ds = ImageDataset(args.data_dir, args.height, args.width, args.sample_every, args.seq_length)
    tfds = convert2tfdataset(ds, DataType.TRAIN, randomize=True)

    tfds = tfds.map(preprocess, num_parallel_calls=multiprocessing.cpu_count())
    tfds = tfds.map(get_augment_fn(args.height, args.width, args.seq_length, NUM_SCALES), num_parallel_calls=multiprocessing.cpu_count())

    # batch, cache, repeat, shuffle, prefetch
    tfds = tfds.batch(args.batch_size)
    tfds = tfds.cache().repeat().shuffle(args.shuffle_buffer_size).prefetch(tf.data.experimental.AUTOTUNE)

    steps_per_epoch = ds.num_examples(DataType.TRAIN) // args.batch_size

    # create the model and train
    model = vid2depth(args.height, args.width, seq_length=args.seq_length, l2=args.weight_norm)

    train(tfds, model, steps_per_epoch=steps_per_epoch, logdir=args.logdir, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.learning_rate, beta1=args.beta1, summary_freq=args.summary_freq,
          reconstr_weight=args.reconstr_weight, ssim_weight=args.ssim_weight, smooth_weight=args.smooth_weight,
          height=args.height, width=args.width, seq_length=args.seq_length, num_scales=NUM_SCALES, l2=args.weight_norm)


def main():
    args = get_args()
    train_model(args)


if __name__ == "__main__":
    main()
