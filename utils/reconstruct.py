import sys
import os
PROJ_ROOT = os.path.abspath(__file__)[:-20]
sys.path.append(PROJ_ROOT)
import tensorflow as tf
from dataloader import CATEGORY_MAP
import scipy.io
from utils import visualization
import importlib
from utils.pick import configure_gpu, get_pred_label
from utils.swap import get_saved_dir
import numpy as np
from tensorflow.keras.utils import Progbar
from multiprocessing import Process
import multiprocessing as mp
from utils.stack_plot import stack_reconstruction_plot
import argparse


def reconstruct(ori_model_p2,
                ori_model_p3,
                notkeepC_model_p2,
                notkeepC_model_p3,
                keepC_model_p2,
                keepC_model_p3,
                hash_codes,
                category,
                H_crop_factor=0.2,
                W_crop_factor=0.5,
                H_shift=15,
                W_shift=50,
                H=32,
                W=32,
                D=32,
                C=1,
                which_gpu=0):

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    configure_gpu(which_gpu)
    saved_dir = get_saved_dir(PROJ_ROOT, 'reconstruct')

    unlabeled_shape_list = list()
    pb1 = Progbar(len(hash_codes))
    print('Saving gt, please wait...')
    for i, code in enumerate(hash_codes):
        unlabeled_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category], code, 'object_unlabeled.mat')
        unlabeled_shape = scipy.io.loadmat(unlabeled_path)['data']
        visualization.save_visualized_img(unlabeled_shape, os.path.join(saved_dir, f'gt_{i}_{code}.png'), is_unlabeled_gt=True)
        unlabeled_shape = unlabeled_shape[..., np.newaxis]
        unlabeled_shape_list.append(unlabeled_shape)
        pb1.update(i+1)

    print('Saving reconstructed images, please wait...')
    p1 = Process(target=save_reconstruction_images, args=(ori_model_p2, unlabeled_shape_list, H, W, D, C, saved_dir, 'recon_ori_p2', hash_codes))
    p2 = Process(target=save_reconstruction_images, args=(ori_model_p3, unlabeled_shape_list, H, W, D, C, saved_dir, 'recon_ori_p3', hash_codes))
    p3 = Process(target=save_reconstruction_images, args=(notkeepC_model_p2, unlabeled_shape_list, H, W, D, C, saved_dir, 'recon_notkeepC_p2', hash_codes))
    p4 = Process(target=save_reconstruction_images, args=(notkeepC_model_p3, unlabeled_shape_list, H, W, D, C, saved_dir, 'recon_notkeepC_p3', hash_codes))
    p5 = Process(target=save_reconstruction_images, args=(keepC_model_p2, unlabeled_shape_list, H, W, D, C, saved_dir, 'recon_keepC_p2', hash_codes))
    p6 = Process(target=save_reconstruction_images, args=(keepC_model_p3, unlabeled_shape_list, H, W, D, C, saved_dir, 'recon_keepC_p3', hash_codes))

    mp.set_start_method('spawn')
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()

    print('Stacking all images together, please wait...')
    stack_reconstruction_plot(saved_dir, H_crop_factor, W_crop_factor, H_shift, W_shift)
    print(f'Done! All images are saved in {saved_dir}')


def save_reconstruction_images(path, unlabeled_shape_list, H, W, D, C, saved_dir, prefix, hash_codes):
    predictions = get_prediction(path, unlabeled_shape_list, H, W, D, C)
    for i, pred in enumerate(predictions):
        pred = get_pred_label(tf.squeeze(tf.where(pred > 0.5, 1., 0.)))
        visualization.save_visualized_img(pred, os.path.join(saved_dir, f'{prefix}_{i}_{hash_codes[i]}.png'))


def get_prediction(path, unlabeled_shape_list, H, W, D, C):
    unlabeled_shape = tf.cast(tf.stack(unlabeled_shape_list, axis=0), dtype=tf.float32)
    model = importlib.import_module(f"results.{path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{path.split('/')[-3]}.hparam")
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], hparam.hparam['bce_weight_shape'],
                           3, hparam.hparam['use_attention'], hparam.hparam['keep_channel'],
                           hparam.hparam['use_ac_loss'], hparam.hparam['which_layer'], hparam.hparam['num_blocks'],
                           hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(path, by_name=True)
    return my_model(unlabeled_shape, training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ori_model_p2', help='path of the ori p2 model')
    parser.add_argument('ori_model_p3', help='path of the ori p3 model')
    parser.add_argument('notkeepC_model_p2', help='path of the notkeepC p2 model')
    parser.add_argument('notkeepC_model_p3', help='path of the notkeepC p3 model')
    parser.add_argument('keepC_model_p2', help='path of the keepC p2 model')
    parser.add_argument('keepC_model_p3', help='path of the keepC p3 model')
    parser.add_argument('--hash_codes', nargs='+', help='hash code of the shapes to be reconstructed', default=['1b7ba5484399d36bc5e50b867ca2d0b9', '9d7d7607e1ba099bd98e59dfd5823115',
                                                                                                                '5b9ebc70e9a79b69c77d45d65dc3714', '88aec853dcb10d526efa145e9f4a2693',
                                                                                                                '5893038d979ce1bb725c7e2164996f48', '96929c12a4a6b15a492d9da2668ec34c',
                                                                                                                '2207db2fa2e4cc4579b3e1be3524f72f'],)
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('--H_crop_factor', default=0.2, help='Percentage to crop empty spcae of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_crop_factor', default=0.5, help='Percentage to crop empty spcae of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('--H_shift', default=15, help='How many pixels to be shifted for the cropping of every single image in H direction. Only valid when save_img is True')
    parser.add_argument('--W_shift', default=55, help='How many pixels to be shifted for the cropping of every single image in W direction. Only valid when save_img is True')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    args = parser.parse_args()

    reconstruct(ori_model_p2=args.ori_model_p2,
                ori_model_p3=args.ori_model_p3,
                notkeepC_model_p2=args.notkeepC_model_p2,
                notkeepC_model_p3=args.notkeepC_model_p3,
                keepC_model_p2=args.keepC_model_p2,
                keepC_model_p3=args.keepC_model_p3,
                hash_codes=args.hash_codes,
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
