import tensorflow as tf


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
