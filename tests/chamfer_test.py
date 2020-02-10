import urllib3
import open3d as o3d
import numpy as np
import tensorflow as tf
import tqdm

from vid2depth.losses import chamfer_distance
from vid2depth.warping import _egomotion_vec2mat
from tensorflow.keras.models import Model
from tensorflow.keras import Input


def get_pcd():
    url = "https://raw.githubusercontent.com/UnaNancyOwen/Tutorials/master/tutorials/interactive_icp/monkey.ply"
    content = urllib3.PoolManager().request("GET", url).data
    with open("monkey.ply", 'wb') as w:
        w.write(content)
    return o3d.io.read_point_cloud("monkey.ply")


def create_simple_model(B, N2, debug=False):

    tv = tf.Variable(tf.zeros([B, 3]), trainable=True, name='translation_vector')
    rv = tf.Variable(tf.zeros([B, 3]), trainable=True, name='rotation_vector')

    @tf.function
    def model(inputs):
        # norm to radians
        #rotation_vector = 2.*np.pi*tf.sigmoid(rotation_vector)
        rotation_vector = np.pi + np.pi * tf.tanh(rv)
        rotation_vector = tf.cast(rotation_vector, tf.float32)

        move_vector = tf.concat([tv, rotation_vector], axis=1)
        move_transform = tf.cast(_egomotion_vec2mat(move_vector, B), tf.float32)

        homo_target = tf.concat([inputs, tf.ones((B, N2, 1), tf.float32)], axis=2)

        # transform first point cloud with move transformation
        transformed_pc2 = tf.matmul(homo_target, move_transform, transpose_b=True)

        # normalize homogenous coordinates to euclidian coordinates
        transformed_pc2 = transformed_pc2 / (1e-3 + tf.tile(tf.expand_dims(transformed_pc2[:, :, 3], 2), [1, 1, 4]))
        transformed_pc2 = transformed_pc2[:, :, :3]

        if debug:
            tf.print('move_vector', move_vector.get_shape().as_list(), B)
            tf.print('move_transform', move_transform.get_shape().as_list(), B)
            tf.print('transformed_pc2', transformed_pc2.get_shape().as_list())
        return transformed_pc2

    return model, [tv, rv]


def chamfer_test(debug=True):

    pcd = get_pcd()
    pc1 = np.asarray(pcd.points)
    pc1 = np.expand_dims(pc1, axis=0)
    print('pc1', pc1.shape, pc1.dtype, pc1.min(), pc1.max())

    # pcd.points = o3d.utility.Vector3dVector(pc1[0])
    pcd.paint_uniform_color([1, 0.706, 0])
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    ## random add noise to the target point cloud
    pc2 = pc1 + np.random.uniform(-000.1, 000.1, [pc1.shape[1], 3])

    ## randomly move second point cloud
    random_offset_vector = np.array([-.01, 0.01, 1.7])
    pc2 += random_offset_vector

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2[0])
    pcd2.paint_uniform_color([1, 0, 0.306])

    B, N1, _ = pc1.shape
    _, N2, _ = pc2.shape

    pc1_v = tf.convert_to_tensor(pc1, tf.float32)
    @tf.function
    def grad(model, variables, x):
        with tf.GradientTape() as tape:
            output = model(x)
            loss_value = chamfer_distance(pc1_v, output)

        gradient = tape.gradient(loss_value, variables)
        return loss_value, gradient

    N = 1000
    lr = 1e-3
    acceptance_loss = 1e-5
    solution_found = False
    x = tf.convert_to_tensor(pc2, tf.float32)

    # inputs = Input(pc2.shape[1:])
    model, variables = create_simple_model(B, N2)

    with tqdm.tqdm(total=N) as tq:
        for i in range(N):
            opt = tf.optimizers.Adam(lr=lr)
            loss_value, grads = grad(model, variables, x)
            opt.apply_gradients(zip(grads, variables))

            if np.min(loss_value) < acceptance_loss:
                best_solution = np.argmin(loss_value)
                solution_found = True
                print('found acceptable loss after %i steps with solution %i! :)' % (i, best_solution))

            if (i > 0 and i % 500 == 0) or solution_found:
                pcd3 = o3d.geometry.PointCloud()
                s = 0
                if solution_found:
                    s = best_solution
                pcd3.points = o3d.utility.Vector3dVector(model(x)[s, :, :])
                pcd3.paint_uniform_color([0, 1, 0.7])
                # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
                o3d.visualization.draw_geometries([pcd, pcd2, pcd3])
            tq.update(1)
            tq.set_postfix(loss=loss_value.numpy())
            if solution_found or i == N:
                break

    print("found solution: ", solution_found)
    print('final result: translation %s, rotation %s' % (variables[0].numpy().tolist(), variables[1].numpy().tolist()))


if __name__ == '__main__':
    chamfer_test()
