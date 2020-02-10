from ..models import vid2depth
from ..losses import depth_smoothness, ssim
from ..warping import inverse_warp
from ..settings import logger

import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D
import tqdm
import os
import argparse


def unpack_image_batches(image_seq, batch_size, height, width, seq_length):
    """[B, h, w * seq_length, 3] -> [B, h, w, 3 * seq_length]."""
    with tf.name_scope('unpack_images'):
        image_list = [
            image_seq[:, :, i * width:(i + 1) * width, :]
            for i in range(seq_length)
        ]
        image_stack = tf.concat(image_list, axis=3)
        image_stack.set_shape([batch_size, height, width, seq_length * 3])
    return image_stack


def build_summaries(output, losses, loss_data, seq_length, num_scales, step):
    """
      loss_data = {
            "scaled_images": scaled_images,
            "warped_images": warped_masks,
            "warped_masks": warped_masks,
            "warped_errors": warped_errors,
            "ssim_errors": ssim_errors
        }

    """
    tf.summary.experimental.set_step(step)
    for key, loss in losses.items():
        tf.summary.scalar('loss/%s' % key, loss)

    egomotion = output['egomotion']
    for i in range(seq_length - 1):
        tf.summary.histogram('ego/tx%d' % i, egomotion[:, i, 0])
        tf.summary.histogram('ego/ty%d' % i, egomotion[:, i, 1])
        tf.summary.histogram('ego/tz%d' % i, egomotion[:, i, 2])
        tf.summary.histogram('ego/rx%d' % i, egomotion[:, i, 3])
        tf.summary.histogram('ego/ry%d' % i, egomotion[:, i, 4])
        tf.summary.histogram('ego/rz%d' % i, egomotion[:, i, 5])

    for s in range(num_scales):
        for i in range(seq_length):
            scaled_image = loss_data['scaled_images'][s][:, :, :, 3 * i:3 * (i + 1)]
            tf.summary.image('scale%d/image%d' % (s, i), scaled_image)

            if i in output['depths']:
                tf.summary.histogram('scale%d/depth%d' % (s, i), output['depths'][i][s])
                tf.summary.histogram('scale%d/disp%d' % (s, i), output['disparities'][i][s])
                tf.summary.image('scale%d/disparity%d' % (s, i), output['disparities'][i][s])

        for key in loss_data['warped_images'][s]:

            tf.summary.image('scale%d/warped_image%s' % (s, key), loss_data['warped_images'][s][key])
            tf.summary.image('scale%d/warp_mask%s' % (s, key), loss_data['warped_masks'][s][key])
            tf.summary.image('scale%d/warp_error%s' % (s, key), loss_data['warped_errors'][s][key])

            if 'ssim' in losses:
                tf.summary.image('scale%d/ssim_error%s' % (s, key), loss_data['ssim_errors'][s][key])


def loss_fn(x, output, reconstr_weight=0.85, ssim_weight=0.15, smooth_weight=0.05,
            batch_size=4, height=128, width=416, seq_length=3, l2=0.05, num_scales=4):

    # print(output.keys())
    image_stack, intrinsic_mat, intrinsic_mat_inv = x

    # build losses
    scaled_images = [{} for _ in range(num_scales)]
    warped_images = [{} for _ in range(num_scales)]
    warped_masks = [{} for _ in range(num_scales)]
    warped_errors = [{} for _ in range(num_scales)]
    ssim_errors = [{} for _ in range(num_scales)]

    losses = {
        "smooth": 0.0,
        "reconstr": 0.0,
        "ssim": 0.0,
    }

    # compute losses at each scale
    # print('scales: ', num_scales)
    for s in range(num_scales):
        # print("compute loss at scape %d" % s)
        height_s = int(height / (2**s))
        width_s = int(width / (2**s))

        scaled_images[s] = tf.image.resize(image_stack, [height_s, width_s], tf.image.ResizeMethod.AREA)

        # smoothness
        if smooth_weight > 0:
            for i in range(seq_length):
                losses['smooth'] += 1.0 / (2**s) * depth_smoothness(output['disparities'][i][s], scaled_images[s][:, :, :, 3 * i:3 * (i + 1)])

        for i in range(seq_length):
            for j in range(seq_length):
                # Only consider adjacent frames.
                if i == j or abs(i - j) != 1:
                    continue

                source = scaled_images[s][:, :, :, 3 * i:3 * (i + 1)]
                target = scaled_images[s][:, :, :, 3 * j:3 * (j + 1)]
                target_depth = output['depths'][j][s]
                key = '%d-%d' % (i, j)

                # Extract ego-motion from i to j
                egomotion_index = min(i, j)
                egomotion_mult = 1

                if i > j:
                    # Need to inverse egomotion when going back in sequence.
                    egomotion_mult *= -1

                egomotion = egomotion_mult * output['egomotion'][:, egomotion_index, :]

                # Inverse warp the source image to the target image frame for
                # photometric consistency loss.
                wimage, wmask = inverse_warp(source,
                                             target_depth,
                                             egomotion,
                                             intrinsic_mat[:, s, :, :],
                                             intrinsic_mat_inv[:, s, :, :])

                warped_images[s][key] = wimage
                warped_masks[s][key] = wmask

                # Reconstruction loss.
                warped_errors[s][key] = tf.abs(warped_images[s][key] - target)
                losses['reconstr'] += tf.reduce_mean(warped_errors[s][key] * warped_masks[s][key])

                # SSIM.
                if ssim_weight > 0:
                    ssim_errors[s][key] = ssim(warped_images[s][key], target)

                    # TODO(rezama): This should be min_pool2d().
                    ssim_mask = AveragePooling2D(3, 1, 'VALID')(warped_masks[s][key])
                    losses['ssim'] += tf.reduce_mean(ssim_errors[s][key] * ssim_mask)

    total_loss = losses['reconstr'] * reconstr_weight
    total_loss += losses['ssim'] * ssim_weight
    total_loss += losses['smooth'] * smooth_weight

    losses['total'] = total_loss

    loss_data = {
        "scaled_images": scaled_images,
        "warped_images": warped_masks,
        "warped_masks": warped_masks,
        "warped_errors": warped_errors,
        "ssim_errors": ssim_errors
    }
    return total_loss, losses, loss_data


def test_model(ds, model):

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


def get_args():
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

    return parser.parse_args()


def main(args=None):
    import multiprocessing

    from ..datasets.image import ImageDataset, DataType
    from ..datasets.utils import convert2tfdataset
    from ..augment import preprocess, get_augment_fn

    if args is None:
        args = get_args()

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


if __name__ == "__main__":
    main()
