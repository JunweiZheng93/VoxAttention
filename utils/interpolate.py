import tensorflow as tf
import sys
import os
PROJ_ROOT = os.path.abspath(__file__)[:-21]
sys.path.append(PROJ_ROOT)
from utils.pick import configure_gpu, get_pred_label
from utils.swap import get_saved_dir
from utils.dataloader import CATEGORY_MAP
import importlib
import scipy.io
import numpy as np
from utils import visualization
import argparse
from utils import stack_plot
from tensorflow.keras.utils import Progbar


def interpolation(model_path,
                  shape1,
                  shape2,
                  category,
                  H_crop_factor=0.2,
                  W_crop_factor=0.55,
                  H_shift=15,
                  W_shift=40,
                  H=32,
                  W=32,
                  D=32,
                  C=1,
                  which_gpu=0):

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    configure_gpu(which_gpu)
    saved_dir = get_saved_dir(PROJ_ROOT, 'interpolate')

    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{model_path.split('/')[-3]}.hparam")

    # load weights and warm up model
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], hparam.hparam['bce_weight_shape'],
                           3, hparam.hparam['use_attention'], hparam.hparam['keep_channel'],
                           hparam.hparam['use_ac_loss'], hparam.hparam['which_layer'], hparam.hparam['num_blocks'],
                           hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    # get hash code of the shapes
    hash_codes = list()
    for first, second in zip(shape1, shape2):
        hash_codes.append((first, second))

    # get unlabeled shapes and labeled shapes
    unlabeled_shape_list = list()
    labeled_shape_list = list()
    for code1, code2 in hash_codes:
        unlabeled_path1 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code1, 'object_unlabeled.mat')
        unlabeled_path2 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code2, 'object_unlabeled.mat')
        labeled_path1 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code1, 'object_labeled.mat')
        labeled_path2 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code2, 'object_labeled.mat')
        unlabeled_shape1 = scipy.io.loadmat(unlabeled_path1)['data'][..., np.newaxis]
        unlabeled_shape2 = scipy.io.loadmat(unlabeled_path2)['data'][..., np.newaxis]
        labeled_shape1 = scipy.io.loadmat(labeled_path1)['data']
        labeled_shape2 = scipy.io.loadmat(labeled_path2)['data']
        unlabeled_shape_list.append(tf.cast(tf.stack((unlabeled_shape1, unlabeled_shape2), axis=0), dtype=tf.float32))
        labeled_shape_list.append([labeled_shape1, labeled_shape2])

    pb = Progbar(len(labeled_shape_list))
    print('Saving images, please wait...')
    for count, (unlabeled_shape, labeled_shape, hash_code) in enumerate(zip(unlabeled_shape_list, labeled_shape_list, hash_codes)):
        shape_latent = interpolate_latent(my_model, unlabeled_shape)

        # get reconstruction and interpolation reconstruction
        fake_input = 0
        shape_outputs = my_model(fake_input, training=False, decomposer_output=shape_latent)

        visualization.save_visualized_img(labeled_shape[0], os.path.join(saved_dir, f'gt_{count}0_{hash_code[0]}.png'))
        visualization.save_visualized_img(labeled_shape[1], os.path.join(saved_dir, f'gt_{count}1_{hash_code[1]}.png'))

        for i, shape_output in enumerate(shape_outputs):
            shape_output = tf.squeeze(tf.where(shape_output > 0.5, 1., 0.))
            shape_output = get_pred_label(shape_output)
            visualization.save_visualized_img(shape_output, os.path.join(saved_dir, f'interpolate_{count}{i}_{hash_code[0]}.png'))
        pb.update(count+1)

    print('Stacking all images together, please wait...')
    stack_plot.stack_interpolation_plot(saved_dir, H_crop_factor=H_crop_factor, W_crop_factor=W_crop_factor, H_shift=H_shift, W_shift=W_shift)
    print(f'Done! All images are saved in {saved_dir}')


def interpolate_latent(my_model, unlabeled_shape):

    full_shape_latent_list = list()

    # get latent representation
    latent = my_model.decomposer(unlabeled_shape, training=False)

    # get latent difference
    latent1, latent2 = tf.unstack(latent, axis=0)
    latent_array1 = latent1.numpy()
    latent_array2 = latent2.numpy()
    latent_diff = latent_array2 - latent_array1

    # interpolate latent
    for i in range(8):
        full_shape_latent_list.append(latent_array1 + i / 7 * latent_diff)
    full_shape_latent = tf.cast(tf.stack(full_shape_latent_list, axis=0), dtype=tf.float32)
    return full_shape_latent


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-s1', '--shape1', nargs='+', default=['96929c12a4a6b15a492d9da2668ec34c', '2a56e3e2a6505ec492d9da2668ec34c'], help='hash code of the first full shape.')
    parser.add_argument('-s2', '--shape2', nargs='+', default=['3c408a4ad6d57c3651bc6269fcd1b4c0', '88aec853dcb10d526efa145e9f4a2693'], help='hash code of the second full shape.')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('--H_crop_factor', default=0.2, help='Percentage to crop empty spcae of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_crop_factor', default=0.55, help='Percentage to crop empty spcae of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('--H_shift', default=15, help='How many pixels to be shifted for the cropping of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_shift', default=40, help='How many pixels to be shifted for the cropping of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    args = parser.parse_args()

    interpolation(model_path=args.model_path,
                  shape1=args.shape1,
                  shape2=args.shape2,
                  category=args.category,
                  H_crop_factor=float(args.H_crop_factor),
                  W_crop_factor=float(args.W_crop_factor),
                  H_shift=int(args.H_shift),
                  W_shift=int(args.W_shift),
                  H=int(args.H),
                  W=int(args.W),
                  D=int(args.D),
                  C=int(args.C),
                  which_gpu=int(args.gpu))
