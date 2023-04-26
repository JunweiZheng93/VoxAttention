import tensorflow as tf
import sys
import os
PROJ_ROOT = os.path.abspath(__file__)[:-14]
sys.path.append(PROJ_ROOT)
from utils.pick import configure_gpu, get_pred_label
from utils.dataloader import CATEGORY_MAP
import importlib
import scipy.io
import numpy as np
from copy import deepcopy
from utils import visualization
import argparse
from utils import stack_plot
from tensorflow.keras.utils import Progbar
from multiprocessing import Process
import multiprocessing as mp


def swap(ori_model_path,
         attention_model_path,
         which_part,
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
    saved_dir = get_saved_dir(PROJ_ROOT, 'swap')

    # get hash code of the shapes
    hash_codes = list()
    for first, second in zip(shape1, shape2):
        hash_codes.append((first, second))

    # get unlabeled shapes and labeled shapes
    unlabeled_shape_list = list()
    labeled_shape_list = list()
    pb = Progbar(len(hash_codes))
    print('Saving gt, please wait...')
    for count, (code1, code2) in enumerate(hash_codes):
        unlabeled_path1 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code1, 'object_unlabeled.mat')
        unlabeled_path2 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code2, 'object_unlabeled.mat')
        labeled_path1 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code1, 'object_labeled.mat')
        labeled_path2 = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code2, 'object_labeled.mat')
        unlabeled_shape1 = scipy.io.loadmat(unlabeled_path1)['data'][..., np.newaxis]
        unlabeled_shape2 = scipy.io.loadmat(unlabeled_path2)['data'][..., np.newaxis]
        labeled_shape1 = scipy.io.loadmat(labeled_path1)['data']
        labeled_shape2 = scipy.io.loadmat(labeled_path2)['data']
        visualization.save_visualized_img(labeled_shape1, os.path.join(saved_dir, f'gt_{count}0_{code1}.png'))
        visualization.save_visualized_img(labeled_shape2, os.path.join(saved_dir, f'gt_{count}1_{code2}.png'))
        unlabeled_shape_list.append([unlabeled_shape1, unlabeled_shape2])
        labeled_shape_list.append([labeled_shape1, labeled_shape2])
        pb.update(count+1)

    print('Saving swapped images, please wait...')
    p1 = Process(target=save_swap_images, args=(ori_model_path, unlabeled_shape_list, hash_codes, which_part, H, W, D, C, saved_dir, 'ori'))
    p2 = Process(target=save_swap_images, args=(attention_model_path, unlabeled_shape_list, hash_codes, which_part, H, W, D, C, saved_dir, 'attention'))
    mp.set_start_method('spawn')
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print('Stacking all images together, please wait...')
    stack_plot.stack_swapped_plot(saved_dir, H_crop_factor=H_crop_factor, W_crop_factor=W_crop_factor, H_shift=H_shift, W_shift=W_shift)
    print(f'Done! All images are saved in {saved_dir}')


def save_swap_images(path, unlabeled_shape_list, hash_codes, which_part, H, W, D, C, save_dir, prefix):

    # load weights and warm up model
    model = importlib.import_module(f"results.{path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{path.split('/')[-3]}.hparam")
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'],
                           hparam.hparam['bce_weight_shape'],
                           3, hparam.hparam['use_attention'], hparam.hparam['keep_channel'],
                           hparam.hparam['use_ac_loss'], hparam.hparam['which_layer'], hparam.hparam['num_blocks'],
                           hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(path, by_name=True)

    for count, (unlabeled_shape, hash_code, which) in enumerate(zip(unlabeled_shape_list, hash_codes, which_part)):
        unlabeled_shape = tf.cast(tf.stack(unlabeled_shape, axis=0), dtype=tf.float32)
        latent, swapped_latent = swap_latent(my_model, unlabeled_shape, int(which))
        # get reconstruction and swapped reconstruction
        fake_input = 0
        outputs = my_model(fake_input, training=False, decomposer_output=latent)
        swapped_outputs = my_model(fake_input, training=False, decomposer_output=swapped_latent)
        for i, (output, swapped_output) in enumerate(zip(outputs, swapped_outputs)):
            output = get_pred_label(tf.squeeze(tf.where(output > 0.5, 1., 0.)))
            swapped_output = get_pred_label(tf.squeeze(tf.where(swapped_output > 0.5, 1., 0.)))
            visualization.save_visualized_img(output, os.path.join(save_dir, f'recon_{prefix}_{count}{i}_{hash_code[i]}.png'))
            visualization.save_visualized_img(swapped_output, os.path.join(save_dir, f'swap_{prefix}_{count}{i}_{hash_code[i]}.png'))


def swap_latent(my_model, unlabeled_shape, which_part):

    # get latent representation
    latent = my_model.decomposer(unlabeled_shape, training=False)

    # swap latent representation
    latent1, latent2 = tf.unstack(latent, axis=0)
    latent_array1 = latent1.numpy()
    latent_array2 = latent2.numpy()
    part_latent = deepcopy(latent_array1[which_part - 1])
    latent_array1[which_part - 1] = latent_array2[which_part - 1]
    latent_array2[which_part - 1] = part_latent
    swapped_latent = tf.cast(tf.stack([latent_array1, latent_array2], axis=0), dtype=tf.float32)
    return latent, swapped_latent


def get_saved_dir(proj_root, mode):
    base_dir = os.path.join(proj_root, 'results', 'figures', mode)
    saved_dir = os.path.join(proj_root, 'results', 'figures', mode)
    count = 1
    while True:
        if os.path.exists(saved_dir):
            saved_dir = f'{base_dir}_{count}'
            count += 1
            continue
        else:
            os.makedirs(saved_dir)
            return saved_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('ori_model_path', help='path of the ori model')
    parser.add_argument('attention_model_path', help='path of the attention model')
    parser.add_argument('-w', '--which_part', default=[3, 1], nargs='+', help='which part to be swapped')
    parser.add_argument('-s1', '--shape1', default=['975ea4be01c7488611bc8e8361bc5303', '9d7d7607e1ba099bd98e59dfd5823115'], nargs='+', help='hash code of the first full shape.')
    parser.add_argument('-s2', '--shape2', default=['297d3e472bf3198fb99cbd993f914184', '1bbe463ba96415aff1783a44a88d6274'], nargs='+', help='hash code of the second full shape.')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('--H_crop_factor', default=0.2, help='Percentage to crop empty spcae of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_crop_factor', default=0.5, help='Percentage to crop empty spcae of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('--H_shift', default=15, help='How many pixels to be shifted for the cropping of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_shift', default=40, help='How many pixels to be shifted for the cropping of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    args = parser.parse_args()

    swap(ori_model_path=args.ori_model_path,
         attention_model_path=args.attention_model_path,
         which_part=args.which_part,
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
