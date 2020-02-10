import tensorflow as tf


def unpack_images(image_seq, height, width, seq_length):
    """[h, w * seq_length, 3] -> [h, w, 3 * seq_length]."""
    with tf.name_scope('unpack_images'):
        image_list = [
            image_seq[:, i * width:(i + 1) * width, :]
            for i in range(seq_length)
        ]
        image_stack = tf.concat(image_list, axis=2)
        image_stack.set_shape([height, width, seq_length * 3])
    return image_stack


def augment_image_colorspace(image_seq):
    """Apply data augmentation to inputs."""
    # Randomly shift gamma.
    random_gamma = tf.random.uniform([], 0.8, 1.2)
    image_seq_aug = image_seq**random_gamma
    # Randomly shift brightness.
    random_brightness = tf.random.uniform([], 0.5, 2.0)
    image_seq_aug *= random_brightness
    # Randomly shift color.
    random_colors = tf.random.uniform([3], 0.8, 1.2)
    white = tf.ones([tf.shape(image_seq)[0], tf.shape(image_seq)[1]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
    image_seq_aug *= color_image
    # Saturate.
    image_seq_aug = tf.clip_by_value(image_seq_aug, 0, 1)
    return image_seq_aug


def make_intrinsics_matrix(fx, fy, cx, cy):
    r1 = tf.stack([fx, 0, cx])
    r2 = tf.stack([0, fy, cy])
    r3 = tf.constant([0., 0., 1.])
    intrinsics = tf.stack([r1, r2, r3])
    return intrinsics


def get_multi_scale_intrinsics(intrinsics, num_scales):
    """Returns multiple intrinsic matrices for different scales."""
    intrinsics_multi_scale = []

    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[0, 0] / (2**s)
        fy = intrinsics[1, 1] / (2**s)
        cx = intrinsics[0, 2] / (2**s)
        cy = intrinsics[1, 2] / (2**s)
        m = make_intrinsics_matrix(fx, fy, cx, cy)
        intrinsics_multi_scale.append(m)

    intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)
    return intrinsics_multi_scale


def scale_randomly(im, intrinsics):
    """Scales image and adjust intrinsics accordingly."""
    in_h, in_w, _ = im.get_shape().as_list()
    scaling = tf.random.uniform([2], 1, 1.15)
    x_scaling = scaling[0]
    y_scaling = scaling[1]
    out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
    out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
    # Add batch.
    im = tf.expand_dims(im, 0)
    im = tf.image.resize(im, [out_h, out_w], tf.image.ResizeMethod.AREA)
    im = im[0]
    fx = intrinsics[0, 0] * x_scaling
    fy = intrinsics[1, 1] * y_scaling
    cx = intrinsics[0, 2] * x_scaling
    cy = intrinsics[1, 2] * y_scaling
    intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
    return im, intrinsics


def crop_randomly(im, intrinsics, out_h, out_w):
    """Crops image and adjust intrinsics accordingly."""
    # batch_size, in_h, in_w, _ = im.get_shape().as_list()
    in_h, in_w, _ = tf.unstack(tf.shape(im))
    offset_y = tf.random.uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
    offset_x = tf.random.uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
    im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
    cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
    intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
    return im, intrinsics


def augment_images_scale_crop(im, intrinsics, out_h, out_w):
    """Randomly scales and crops image."""

    im, intrinsics = scale_randomly(im, intrinsics)
    im, intrinsics = crop_randomly(im, intrinsics, out_h, out_w)
    return im, intrinsics


def preprocess(image_seq, intrinsics):
    image_seq = tf.image.convert_image_dtype(image_seq, tf.float32)
    return image_seq, intrinsics


def get_augment_fn(height, width, seq_length, num_scales):

    def augment(image_seq, intrinsics):
        image_seq = augment_image_colorspace(image_seq)
        image_stack = unpack_images(image_seq, height, width, seq_length)
        image_stack, intrinsics = augment_images_scale_crop(image_stack, intrinsics, height, width)

        # get intrinsics multiscale, get inverse
        intrinsic_mat = get_multi_scale_intrinsics(intrinsics, num_scales)
        intrinsic_mat.set_shape([num_scales, 3, 3])
        intrinsic_mat_inv = tf.linalg.inv(intrinsic_mat)
        intrinsic_mat_inv.set_shape([num_scales, 3, 3])

        return image_stack, intrinsic_mat, intrinsic_mat_inv
    return augment
