from . import warping

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


def reconstruction_loss(warped_image, warped_mask, target):
    error = tf.abs(warped_image - target)
    return tf.reduce_mean(error * warped_mask)


def depth_smoothness_loss(depth, img, s):
    return 1.0 / (2**s) * depth_smoothness(depth, img)


def ssim_loss(x, y):
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


def chamfer_distance(pc1, pc2, norm=2, egomotion=None, debug=False):
    """Computes the Chamfer Distance

    Measures the distance between two point clouds.
    Calculates average minimum distance for each point to the nearest point of the other pointcloud.
    Optional egomotion vector with shape [B,6] that transforms pc1. 

    Arguments:
        pc1 {tf.Tensor} -- pointcloud 1 of shape [B, N, ?]
        pc2 {tf.Tensor} -- pointcloud 2 of shape [B, N, ?]
        norm {int} -- int, if 2 -> mse distance, 1 -> abs distance (default: {2})
        egomotion {tf.Tensor} -- egomotion vector of shape [B, 6] (default: {None})
        debug {bool} -- debug shapes

    Returns:
        {float}: chamfer mean distance
    """
    t1 = tf.convert_to_tensor(pc1)
    t2 = tf.convert_to_tensor(pc2)
    inshape = t1.get_shape().as_list()
    B, N, _ = inshape

    ## if egomotion is supplied, transform pc1 by egomotion
    if egomotion is not None:
        egomotion_mat = warping._egomotion_vec2mat(egomotion, B)

        # euclid -> homo
        t1 = tf.concat([t1, tf.ones([B, N, 1])], axis=2)

        # transform first pc by egomotion
        t1 = tf.transpose(tf.matmul(egomotion_mat, tf.transpose(t1)))

        # homo -> euclid
        t1 = t1 / tf.tile(tf.expand_dims(t1[:, :, 3], 2), [1, 1, 4])
        t1 = t1[:, :, :3]

    mmax = tf.reduce_max([tf.reduce_max(t1), tf.reduce_max(t2)])
    t1 /= mmax  # norm to soft [0,1]
    t2 /= mmax

    B, N1, _ = t1.get_shape().as_list()
    B, N2, _ = t2.get_shape().as_list()

    if debug:
        tf.print('pc1', t1.get_shape().as_list())
        tf.print('pc2', t2.get_shape().as_list())

    ## for each point of pc1 , find shortest euclidian distance to any point of pc2
    # make matrix [n,n] with each element being euclidian distance (i,j)
    # for each line take minimum
    # mean row
    x1, y1, z1 = t1[:, :, 0], t1[:, :, 1], t1[:, :, 2]
    x2, y2, z2 = t2[:, :, 0], t2[:, :, 1], t2[:, :, 2]
    [x1, y1, z1, x2, y2, z2] = [tf.cast(tf.reshape(_a, [B, N1, 1]), tf.float32) for _a in [x1, y1, z1, x2, y2, z2]]

    ## span up big mats
    xx1 = tf.matmul(x1, tf.ones((B, 1, N2), tf.float32))
    yy1 = tf.matmul(y1, tf.ones((B, 1, N2), tf.float32))
    zz1 = tf.matmul(z1, tf.ones((B, 1, N2), tf.float32))
    xx2 = tf.matmul(x2, tf.ones((B, 1, N1), tf.float32))
    yy2 = tf.matmul(y2, tf.ones((B, 1, N1), tf.float32))
    zz2 = tf.matmul(z2, tf.ones((B, 1, N1), tf.float32))

    ## calc differences and distances
    dxx = xx2 - xx1
    dyy = yy2 - yy1
    dzz = zz2 - zz1

    if norm == 2:
        distance_mat = tf.sqrt(1e-7 + dxx * dxx + dyy * dyy + dzz * dzz)
    elif norm == 1:
        distance_mat = tf.abs(dxx) + tf.abs(dyy) + tf.abs(dzz) + 1e-7

    ## calc avg minimum distance for each point
    min_distances = tf.reduce_min(distance_mat, axis=1)
    chamfer = tf.reduce_mean(min_distances, axis=1)

    if debug:
        tf.print('distance_mat', distance_mat.get_shape().as_list())
        tf.print('min_distances', min_distances.get_shape().as_list())
        tf.print('chamfer', chamfer.get_shape().as_list())
    return chamfer


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
                losses['smooth'] += depth_smoothness_loss(output['disparities'][i][s], scaled_images[s][:, :, :, 3 * i:3 * (i + 1)], s)

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
                wimage, wmask = warping.inverse_warp(source,
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
                    ssim_errors[s][key] = ssim_loss(warped_images[s][key], target)

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
