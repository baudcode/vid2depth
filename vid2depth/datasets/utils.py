from .dataset import DataType
from ..settings import logger

import math
import os
import random
import numpy as np
import tensorflow as tf
import multiprocessing
import tqdm


def parallize(f, args, n_processes=None, desc='parallize'):
    if n_processes == None:
        n_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(n_processes) as pool:
        results = [r for r in tqdm.tqdm(pool.imap(f, args), desc=desc, total=len(args))]

    return results


def convert2tfdataset(dataset, data_type, randomize=True):

    def gen():
        indexes = np.arange(dataset.num_examples(data_type))
        if randomize:
            indexes = np.random.permutation(indexes)

        data = dataset.raw()[data_type]
        for idx in indexes:
            example = data[idx]
            try:
                image_stack, intrinsics = dataset.parse_example(example)
            except Exception as e:
                logger.warning("could not parse example %s, error: %s, skipping" % (str(example), str(e)))
                continue

            assert(image_stack.shape[-1] == 3)
            yield image_stack, intrinsics, image_stack.shape

    def map_fn(image_stack, intrinsics, shape):
        image_stack = tf.reshape(image_stack, shape)
        return image_stack, intrinsics

    ds = tf.data.Dataset.from_generator(
        gen, (tf.uint8, tf.float32, tf.int64), ([None, None, None], [3, 3], [3]))
    ds = ds.map(map_fn, num_parallel_calls=multiprocessing.cpu_count())

    return ds


def get_train_test_val_from_list(l, train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.2):
    if shuffle:
        random.shuffle(l, random=rand)

    trainset = l[:int(round(train_split * len(l)))]
    valtestset = l[int(round(train_split * len(l))):]
    testset = valtestset[int(round(val_split * len(valtestset))):]
    valset = valtestset[:int(round(val_split * len(valtestset)))]
    return trainset, testset, valset


def get_split(l, train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.2):
    trainset, testset, valset = get_train_test_val_from_list(
        l, train_split=train_split, val_split=val_split, shuffle=shuffle, rand=rand)
    return {
        DataType.TRAIN: trainset,
        DataType.VAL: valset,
        DataType.TEST: testset
    }


def get_files(directory, extensions=None):
    files = []
    extensions = [ext.lower() for ext in extensions] if extensions else None
    for root, _, filenames in sorted(os.walk(directory)):
        if extensions:
            files += [os.path.join(root, name) for name in sorted(filenames)
                      if any(name.lower().endswith("." + ext) for ext in extensions)]
        else:
            files += list(map(lambda filename: os.path.join(root,
                                                            filename), filenames))

    return sorted(files)


def load_intrinsics(cubelength):
    """Load intrinsics."""
    # https://www.wired.com/2013/05/calculating-the-angular-view-of-an-iphone/
    # https://codeyarns.com/2015/09/08/how-to-compute-intrinsic-camera-matrix-for-a-camera/
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    # # iPhone: These numbers are for images with resolution 720 x 1280.
    # Assuming FOV = 50.9 => fx = (1280 // 2) / math.tan(fov / 2) = 1344.8
    #fx = ( cubelength//2 ) / math.tan(np.pi*45./180.)
    fx = (cubelength // 2) / math.tan((90. / 2.) * np.pi / 180.)
    intrinsics = np.array([[fx, 0, cubelength // 2],
                           [0, fx, cubelength // 2],
                           [0, 0, 1.0]], np.float32)
    return intrinsics


def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out
