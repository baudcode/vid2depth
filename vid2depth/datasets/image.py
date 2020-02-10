from .dataset import DataType, Dataset
from .utils import get_files, get_split, scale_intrinsics

import os
import numpy as np
import imageio
import random
import cv2


def load_intrinsics(cy):
    """Load intrinsics."""
    # https://www.wired.com/2013/05/calculating-the-angular-view-of-an-iphone/
    # https://codeyarns.com/2015/09/08/how-to-compute-intrinsic-camera-matrix-for-a-camera/
    # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
    # # iPhone: These numbers are for images with resolution 720 x 1280.
    # Assuming FOV = 50.9 => fx = (1280 // 2) / math.tan(fov / 2) = 1344.8
    intrinsics = np.array([[1344.8, 0, 1280 // 2],
                           [0, 1344.8, cy],
                           [0, 0, 1.0]])
    return intrinsics


def load_image(path, height, width, crop_type=0):
    """Reads the image and crops it according to first letter of frame_id."""
    img = imageio.imread(path)
    allowed_height = int(img.shape[1] * height / width)
    # Starting height for the middle crop.
    mid_crop_top = int(img.shape[0] / 2 - allowed_height / 2)
    # How much to go up or down to get the other two crops.
    height_var = int(mid_crop_top / 3)

    if crop_type == 0:
        crop_top = mid_crop_top - height_var
        cy = allowed_height / 2 + height_var
    elif crop_type == 1:
        crop_top = mid_crop_top
        cy = allowed_height / 2
    elif crop_type == 2:
        crop_top = mid_crop_top + height_var
        cy = allowed_height / 2 - height_var
    else:
        raise ValueError('Unknown crop_type: %s' % crop_type)

    crop_bottom = crop_top + allowed_height + 1
    return img[crop_top:crop_bottom, :, :], cy


def get_examples(data_dir, seq_length=3, sample_every=6):
    # get examples by video
    examples = []
    for video in filter(lambda x: os.path.isdir(os.path.join(data_dir, x)), os.listdir(data_dir)):
        video_dir = os.path.join(data_dir, video)
        images = get_files(video_dir, extensions=['jpg', 'png'])
        for i in range(0, len(images) - seq_length * sample_every):
            example = []
            for l in range(seq_length):
                example.append(images[i + l * sample_every])
            examples.append(example)

    return examples


def parse_example(example, height, width, crop_type=0):
    images = []

    for path in example:
        image, cy = load_image(path, height, width, crop_type=crop_type)

        image = cv2.resize(image, (width, height))
        images.append(image)

    target = images[len(example) // 2]
    zoom_y = height / target.shape[0]
    zoom_x = width / target.shape[1]

    intrinsics = load_intrinsics(cy)
    intrinsics = scale_intrinsics(intrinsics, zoom_x, zoom_y)
    image_stack = np.concatenate(images, axis=1)

    return image_stack, intrinsics


class ImageDataset(Dataset):

    def __init__(self, cache_dir, height=128, width=416, sample_every=6, seq_length=3):
        super(ImageDataset, self).__init__(cache_dir)
        self.seq_length = seq_length
        self.height = height
        self.width = width
        self.sample_every = sample_every

    def raw(self):
        examples = get_examples(self.cache_dir, seq_length=self.seq_length, sample_every=self.sample_every)
        return get_split(examples)

    def parse_example(self, example):
        crop_type = random.randint(0, 2)
        return parse_example(example, self.height, self.width, crop_type=crop_type)


if __name__ == "__main__":
    from .utils import convert2tfdataset

    ds = ImageDataset('/media/baudcode/Data/datasets/walk/videos/')
    tfds = convert2tfdataset(ds, DataType.TRAIN)

    for image_stack, intrinsics in tfds:
        print(image_stack.shape, intrinsics.shape)
        break
