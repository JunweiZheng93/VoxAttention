import os
import numpy as np
import tensorflow as tf
import math
import scipy.io
from tensorflow.keras.utils import Sequence, Progbar

CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649', 'guitar': '03467517'}
URL_MAP = {'chair': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03001627.zip?inline=false',
           'table': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/04379243.zip?inline=false',
           'airplane': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/02691156.zip?inline=false',
           'lamp': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03636649.zip?inline=false',
           'guitar': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03467517.zip?inline=false'}
PROJ_ROOT = os.path.abspath(__file__)[:-19]


def get_dataset(category='chair', batch_size=32, split_ratio=0.8, max_num_parts=4):

    category_path = download_dataset(category)
    voxel_grid_fp, part_fp, trans_fp = get_fp(category_path)
    num_training_samples = math.ceil(len(voxel_grid_fp) * split_ratio)

    all_voxel_grid = list()
    all_part = list()
    all_trans = list()
    pb = Progbar(len(voxel_grid_fp))
    print('Loading data, please wait...')
    for count, (v_fp, p_fp, t_fp) in enumerate(zip(voxel_grid_fp, part_fp, trans_fp)):
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
                transformations.append(scipy.io.loadmat(os.path.join(dir_name, f'part{i}_trans_matrix.mat'))['data'][:3])
        all_part.append(parts)
        all_trans.append(transformations)
        pb.update(count+1)
    print(f'Total Samples: {len(voxel_grid_fp)} - Training Samples: {num_training_samples} - Test Samples: {len(voxel_grid_fp)-num_training_samples}')

    training_set = Dataset(all_voxel_grid[:num_training_samples], all_part[:num_training_samples],
                           all_trans[:num_training_samples], batch_size)
    test_set = Dataset(all_voxel_grid[num_training_samples:], all_part[num_training_samples:],
                       all_trans[num_training_samples:], batch_size)

    return training_set, test_set


def download_dataset(category):

    # check category
    if category not in list(CATEGORY_MAP.keys()):
        raise ValueError(f'category should be one of chair, table, airplane and lamp. got {category} instead!')

    category_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])
    if not os.path.exists(category_path):
        tf.keras.utils.get_file(f'{CATEGORY_MAP[category]}.zip', URL_MAP[category], cache_dir=PROJ_ROOT, extract=True)
        os.remove(f'{category_path}.zip')
    return category_path


def get_fp(category_fp):
    shape_paths = sorted([os.path.join(category_fp, shape_name) for shape_name in os.listdir(category_fp)])
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
    return voxel_grid_fp, part_fp, trans_fp


class Dataset(Sequence):

    def __init__(self, all_voxel_grid, all_part, all_trans, batch_size):
        self.all_voxel_grid = all_voxel_grid
        self.all_part = all_part
        self.all_trans = all_trans
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.all_voxel_grid) / self.batch_size)

    def __getitem__(self, idx):
        batch_voxel_grid = self.all_voxel_grid[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_part = self.all_part[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_trans = self.all_trans[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.asarray(batch_voxel_grid, dtype='float32'), np.asarray(batch_part, dtype='float32'), np.asarray(batch_trans, dtype='float32')