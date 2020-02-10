import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input

from . import disp, egomotion


def vid2depth(height, width, seq_length=3, l2=0.05):
    submodels = {}

    image_stack = Input((height, width, seq_length * 3))

    ego_model = egomotion.egomotion_model(image_stack)
    submodels['egomotion'] = ego_model

    disps = {}
    depths = {}

    # build depth prediction for every image
    for i in range(seq_length):
        multiscale_disp_model = disp.disp_model(image_stack, i, seq_length, l2=l2)
        submodels['disp%i' % i] = multiscale_disp_model
        disps[i] = multiscale_disp_model.outputs
        depths[i] = [1.0 / o for o in multiscale_disp_model.outputs]

    model = Model(inputs=image_stack, outputs={
        "egomotion": ego_model.outputs[0],
        "disparities": disps,
        "depths": depths
    })
    return model, submodels


if __name__ == "__main__":
    model, submodels = vid2depth(128, 416)
    print(model.output)
