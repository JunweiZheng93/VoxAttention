import sys
import os
PROJ_ROOT = os.path.abspath(__file__)[:-14]
sys.path.append(PROJ_ROOT)
import tensorflow as tf
import importlib
import scipy.io
import numpy as np
import argparse
from utils.dataloader import Dataset
from utils import visualization
import shutil
from tensorflow.keras.utils import Progbar


CHAIRS = ['54e2aa868107826f3dbc2ce6b9d89f11', '5042005e178d164481d0f12b8bf5c990', 'b8e4d2f12e740739b6c7647742d948e',
          '2207db2fa2e4cc4579b3e1be3524f72f', '2a87cf850dca72a4a886e56ff3d54c4', '3c408a4ad6d57c3651bc6269fcd1b4c0',
          '9ab18a33335373b2659dda512294c744', '5b9ebc70e9a79b69c77d45d65dc3714', '1bbe463ba96415aff1783a44a88d6274',
          'b2ded1854643f1451c1b3b2ed8d13bf8', 'dfc9e6a84553253ef91663a74ccd2338', '5893038d979ce1bb725c7e2164996f48',
          '88aec853dcb10d526efa145e9f4a2693', '611f235819b7c26267d783b4714d4324', 'cd9702520ad57689bbc7a6acbd8f058b',
          '2a56e3e2a6505ec492d9da2668ec34c', '5a643c0c638fc2c3ff3a3ae710d23d1e', '96929c12a4a6b15a492d9da2668ec34c',
          '1b7ba5484399d36bc5e50b867ca2d0b9', '2fed64c67552aa689c1db271ad9472a7', '9d7d7607e1ba099bd98e59dfd5823115',
          '875925d42780159ffebad4f49b26ec52', '2025aa3a71f3c468d16ba2cb1292d98a']

AIRPLANES = ['1b7ac690067010e26b7bd17e458d0dcb', '1a04e3eab45ca15dd86060f189eb133', '1deb997079e0b3cd6c1cd53dbc9f7b8e',
             '2b1a867569f9f61a54eefcdc602d4520', '2c1fff0653854166e7a636089598229', '2c64c521c114df40e51f766854841067',
             '2c97e6b2c92913cac1ccec171a275967', '2d01483c696c0a1688be2a30dd556a09', '2e961e38a039f1bc67711f7c205c5b63',
             '3ae96a1e1bb488942296d88107d065f6', '3ad337dcef167024fe6302fece358e4a', '3bad4bd2c944d78391d77854c55fb8fc',
             '3e0561d70c7fd4f51c6e4e20f2b76086', '3feeb5f8ecbfcb4ba8f0518e94fcfb22', '4a837740b388aa45d8ff6111270336a9',
             '4e4128a2d12c818e5f38952c9fdf4604', '4f7814692598ebdc7dadbbeb79fd1fc9', '6dedeb5b87ee318b2154ead1f7ab03aa']

LAMPS = ['5c9f3efb7603bd107c57db67d218d3b9', '4c266f2b866c59e761fef32872c6fa53', '4da9ae6448c860243dfad56d2a4eefcd',
         '5b74e8ee70acba2827d25c76a863dd52', '5b12386df80fe8b0664b3b9b23ddfcbc', '6f5b104f7c1cb4bc636c7e486232cac1',
         '9a244723bfef786294cdfc338037bd95', '9cfefdc1e63a3c2535836c728d324152', '9d340cb226868e39ce4f274577283b16',
         '12d44fd814bc9b40ec2a7a1f5fe7365d', '41c1e411592ecabcb7487183c0e206af', '82c10d98a1aa65e89730cb37c9a5f63b',
         '5116452b7826dfd92548598e855f0844', 'a7b9dc38c80e86fbb13604bbce4eb6a8', 'a911474d0fa043e3cf004563556ddb36',
         'ead77648c9c7dbf8d42b9650f19dd425', 'f274cbff23bda61a85024815e816c655', '1e62d260a8a64b5d8f720345751070e9']

TABLES = ['1a43bd2e53364313f51f77a6d7299806', '1a00aa6b75362cc5b324368d54a7416f', '1b7dd5d16aa6fdc1f716cef24b13c00',
          '1aed00532eb4311049ba300375be3b4', '1c66f97bf8375052c13e020d985215e3', '1d4e22bc8ed400fc368162d385acdaa9',
          '1d53304accfb6fb3c3bd24f986301745', '1f59698c02bd1662dbbc9440457e303e', '2b51c3e9b524ddf560b5fd678a94e9cd',
          '2c3a4ab3efbf12d74ef530b007e93f59', '2ca883ba6a9dc6f68985be89a0ee21a', '2d90a1998eca8778dcfcef693e7ec696',
          '2d7c48df632589a7ad5067eac75a07f7', '2f9c9357bfb89ac1d38913e96bbf2a5d', '3a990272ef4b83ca8d3e8783b997c75',
          '3b3b35ab4412c3263edd525d249a1362', '3ed500a12dfa511ba6040757a0125a99', '6f58b8c1d826a301a97bcacc05204e5c']

GUITAR = ['1a96f73d0929bd4793f0194265a9746c', '1a680e3308f2aac544b2fa2cac0778f5', '1a8512735ed86bc52d7d603c684cb89e',
          '1abe78447898821e93f0194265a9746c', '1b65d2e0c0ed78237e1c85c5c15da7fb', '1c8c6874c0cb9bc73429c1c21d77499d',
          '2c1b6a4fb58b04371a3c749a0d8db8f3', '2c2fc56385be92b493f0194265a9746c', '2c491c5718322fc4849457db85ec22c6',
          '2cbc0faddf227502bbc745a3524d966b', '2dbc73ad4ce7950163e148e250c0340d', '2eba922263fc1580cc010a80df5d3c87',
          '2f1ef453d18cc88d74f1a026675e6353', '2f9d51c98d517ed1b647271c21ec40', '3f94fd316d4f2ca1d57700c05b1862d8',
          '3fba85bfdb125b1b93f0194265a9746c', '4a704919068627caeae5cab1248d1ec6', '4ae5a491c3ffb473462c6cdd250c26bb']


def pick(model_path,
         H=32,
         W=32,
         D=32,
         C=1,
         which_gpu=0):

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    configure_gpu(which_gpu)

    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{model_path.split('/')[-3]}.hparam")

    # load weights and warm up model
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], hparam.hparam['bce_weight_shape'],
                           3, hparam.hparam['use_attention'], hparam.hparam['keep_channel'], hparam.hparam['use_ac_loss'],
                           hparam.hparam['which_layer'], hparam.hparam['num_blocks'], hparam.hparam['num_heads'],
                           hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    test_set, shape_paths = get_dataset(hparam.hparam['category'], 1, hparam.hparam['max_num_parts'])

    saved_dir = os.path.join(PROJ_ROOT, 'results', model_path.split('/')[-3], 'picks')
    if os.path.exists(saved_dir):
        shutil.rmtree(saved_dir)
    os.makedirs(saved_dir)

    pb = Progbar(len(shape_paths))
    print('Saving images, please wait...')
    for count, ((voxel_grid, label, trans), shape_path) in enumerate(zip(test_set, shape_paths)):
        stacked_transformed_parts = my_model(voxel_grid, training=False)
        gt_label_path = os.path.join(shape_path, 'object_labeled.mat')
        gt_label = scipy.io.loadmat(gt_label_path)['data']
        shape_name = gt_label_path.split('/')[-2]
        visualization.save_visualized_img(gt_label, os.path.join(saved_dir, f'{shape_name}_gt.png'))
        if hparam.hparam['training_process'] == 1 or hparam.hparam['training_process'] == '1':
            pred = tf.squeeze(tf.where(my_model.stacked_decoded_parts > 0.5, 1., 0.))
            pred = pred.numpy().astype('uint8')
            for i, part in enumerate(pred):
                visualization.save_visualized_img(part, os.path.join(saved_dir, f'{shape_name}_part{i+1}_recon.png'))
        else:
            pred = tf.squeeze(tf.where(stacked_transformed_parts > 0.5, 1., 0.))
            pred_label = get_pred_label(pred)
            visualization.save_visualized_img(pred_label, os.path.join(saved_dir, f'{shape_name}_shape_recon.png'))
        pb.update(count+1)
    print(f'Done! All images are saved in {saved_dir}')


def get_dataset(category, batch_size, max_num_parts):
    voxel_grid_fp, part_fp, trans_fp, shape_paths = get_fp(category)
    all_voxel_grid = list()
    all_part = list()
    all_trans = list()
    for v_fp, p_fp, t_fp in zip(voxel_grid_fp, part_fp, trans_fp):
        v = scipy.io.loadmat(v_fp)['data'][:, :, :, np.newaxis]
        all_voxel_grid.append(v)
        parts = list()
        transformations = list()
        member_list = [int(each[-5]) for each in p_fp]
        dir_name = os.path.dirname(v_fp)
        for i in range(1, max_num_parts + 1):
            if i not in member_list:
                part = np.zeros_like(v, dtype='uint8')
                parts.append(part)
                transformation = np.zeros((3, 4), dtype='float32')
                transformations.append(transformation)
            else:
                part = scipy.io.loadmat(os.path.join(dir_name, f'part{i}.mat'))['data'][:, :, :, np.newaxis]
                parts.append(part)
                transformations.append(
                    scipy.io.loadmat(os.path.join(dir_name, f'part{i}_trans_matrix.mat'))['data'][:3])
        all_part.append(parts)
        all_trans.append(transformations)
    test_set = Dataset(all_voxel_grid, all_part, all_trans, batch_size)
    return test_set, shape_paths


def get_fp(category):

    if category == 'chair':
        category_fp = os.path.join(PROJ_ROOT, 'datasets', '03001627')
        shape_paths = [os.path.join(category_fp, shape_name) for shape_name in CHAIRS]
    elif category == 'airplane':
        category_fp = os.path.join(PROJ_ROOT, 'datasets', '02691156')
        shape_paths = [os.path.join(category_fp, shape_name) for shape_name in AIRPLANES]
    elif category == 'lamp':
        category_fp = os.path.join(PROJ_ROOT, 'datasets', '03636649')
        shape_paths = [os.path.join(category_fp, shape_name) for shape_name in LAMPS]
    elif category == 'guitar':
        category_fp = os.path.join(PROJ_ROOT, 'datasets', '03467517')
        shape_paths = [os.path.join(category_fp, shape_name) for shape_name in GUITAR]
    elif category == 'table':
        category_fp = os.path.join(PROJ_ROOT, 'datasets', '04379243')
        shape_paths = [os.path.join(category_fp, shape_name) for shape_name in TABLES]
    else:
        raise ValueError('category should be one of chair, table, airplane, guitar and lamp!')

    voxel_grid_fp = list()
    part_fp = list()
    trans_fp = list()
    for shape_path in shape_paths:
        voxel_grid = os.path.join(shape_path, 'object_unlabeled.mat')
        part_list = list()
        trans_list = list()
        all_files = sorted(os.listdir(shape_path))
        for file in all_files:
            if file.startswith('part') and file.endswith('.mat'):
                if file.startswith('part') and file.endswith('trans_matrix.mat'):
                    trans_list.append(os.path.join(shape_path, file))
                    continue
                part_list.append(os.path.join(shape_path, file))
        voxel_grid_fp.append(voxel_grid)
        part_fp.append(part_list)
        trans_fp.append(trans_list)
    return voxel_grid_fp, part_fp, trans_fp, shape_paths


def configure_gpu(which_gpu):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[which_gpu], "GPU")


def get_pred_label(pred):
    code = 0
    for idx, each_part in enumerate(pred):
        code += each_part * 2 ** (idx + 1)
    pred_label = tf.math.floor(tf.experimental.numpy.log2(code + 1))
    pred_label = pred_label.numpy().astype('uint8')
    return pred_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    args = parser.parse_args()

    pick(model_path=args.model_path,
         H=int(args.H),
         W=int(args.W),
         D=int(args.D),
         C=int(args.C),
         which_gpu=int(args.gpu))
