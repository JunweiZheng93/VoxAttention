import math
import numpy as np
import os
import sys
PROJ_ROOT = os.path.abspath(__file__)[:-28]
sys.path.append(PROJ_ROOT)
import scipy.io
import argparse
from utils import binvox_rw
from utils import visualization
from scipy import ndimage
from tensorflow.keras.utils import Progbar


def get_reference_label(pcd_fp, label_fp, resolution=32):
    """
    :param pcd_fp: file path of the point cloud data
    :param label_fp: file path of the corresponding label to the point cloud data
    :param resolution: resolution of the voxel grid that you want to generate
    :return: voxel grid label generated from point cloud label
    """
    pcd = np.genfromtxt(pcd_fp, dtype='float32')
    pcd_label = np.genfromtxt(label_fp, dtype='uint8')
    maxi = np.max(pcd, axis=0)
    mini = np.min(pcd, axis=0)
    voxel_size = np.max(np.ceil((maxi - mini) * 10000 / (resolution - 1)) / 10000)
    translation = np.max(pcd) + 0.5 * voxel_size
    pcd += translation

    voxel_grid_label_dict = dict()
    for point, point_label in zip(pcd, pcd_label):
        idx = point // voxel_size
        voxel_idx = int(idx[0] + idx[1] * resolution + idx[2] * resolution ** 2)
        if voxel_idx not in list(voxel_grid_label_dict.keys()):
            voxel_grid_label_dict[voxel_idx] = [point_label]
        else:
            voxel_grid_label_dict[voxel_idx].append(point_label)

    voxel_grid_label = np.full((resolution, resolution, resolution), 0, dtype='uint8')
    for voxel_idx in list(voxel_grid_label_dict.keys()):
        count_list = voxel_grid_label_dict[voxel_idx]
        voxel_label = max(set(count_list), key=count_list.count)
        idx = [voxel_idx % resolution, voxel_idx // resolution % resolution, voxel_idx // resolution // resolution % resolution]
        voxel_grid_label[idx[0], idx[1], idx[2]] = voxel_label
    return voxel_grid_label


def get_surface_label(voxel_grid, reference_label):
    """
    :param voxel_grid: voxel grid generated from binvox file
    :param reference_label: voxel grid label generated from point cloud label
    :return: surface label for the voxel grid generated from binvox file
    """
    voxel_grid_cord = np.stack(np.where(voxel_grid), axis=1)
    ref_label_cord = np.stack(np.where(reference_label > 0), axis=1)
    surface_label = np.zeros_like(voxel_grid, dtype='uint8')
    for cord in ref_label_cord:
        dist = cord - voxel_grid_cord
        mini_dist_idx = np.argmin(np.linalg.norm(dist, axis=1))
        surface_label_idx = voxel_grid_cord[mini_dist_idx]
        surface_label[surface_label_idx[0], surface_label_idx[1], surface_label_idx[2]] = reference_label[cord[0], cord[1], cord[2]]
    return surface_label


def get_voxel_grid_label(voxel_grid, surface_label, k=5):
    """
    :param voxel_grid: voxel grid generated from binvox file
    :param surface_label: surface label for the voxel grid generated from binvox file
    :param k: knn parameter for calculating the inner voxel label
    :return: voxel grid label (surface and inner voxels are all labeled)
    """
    voxel_grid_cord = np.stack(np.where(voxel_grid), axis=1)
    surface_label_cord = np.stack(np.where(surface_label > 0), axis=1)
    for cord in voxel_grid_cord:
        if list(cord) not in surface_label_cord.tolist():
            candidate_label_list = []
            dist = cord - surface_label_cord
            mini_dist_idx = np.argpartition(np.linalg.norm(dist, axis=1), k)[:k]
            candidate_cord = surface_label_cord[mini_dist_idx]
            for each in candidate_cord:
                candidate_label_list.append(surface_label[each[0], each[1], each[2]])
            voxel_label = max(set(candidate_label_list), key=candidate_label_list.count)
            surface_label[cord[0], cord[1], cord[2]] = voxel_label
            surface_label_cord = np.stack(np.where(surface_label > 0), axis=1)
    return surface_label


def get_seperated_part_and_transformation(voxel_grid_label):
    """
    :param voxel_grid_label: voxel grid label to be seperated
    :return: part_voxel_grid_array in shape (num_parts, H, W, D), transformation_matrix_array in shape (num_parts, 4, 4)
            which_part in shape (num_parts,)
    """
    num_parts = np.max(voxel_grid_label)
    resolution = voxel_grid_label.shape[0]
    voxel_grid_center = np.full((3,), (resolution - 1) // 2)
    part_voxel_grid_list = list()
    transformation_matrix_list = list()
    which_part = list()

    for i in range(1, num_parts+1):
        original_part_voxel_grid = voxel_grid_label == i
        if not original_part_voxel_grid.any():
            continue
        which_part.append(i)
        part_cord = np.stack(np.where(original_part_voxel_grid), axis=1)

        # get scaled and centered part voxel grid
        part_min_cord = np.min(part_cord, axis=0)
        part_max_cord = np.max(part_cord, axis=0)
        part_bbox_hwd = part_max_cord - part_min_cord + 1
        scale_factor = math.floor((resolution / np.max(part_bbox_hwd)) * 100) / 100
        if scale_factor > 2.:
            scale_factor = 2.
        part_bbox = np.full((part_bbox_hwd[0], part_bbox_hwd[1], part_bbox_hwd[2]), 0, dtype='uint8')
        for each_cord in (part_cord - part_min_cord):
            part_bbox[each_cord[0], each_cord[1], each_cord[2]] = 1
        scaled_part_bbox = ndimage.zoom(part_bbox, scale_factor, order=0)
        scaled_part_bbox_center = (np.asarray(scaled_part_bbox.shape) - 1) // 2
        dist = voxel_grid_center - scaled_part_bbox_center
        scaled_centered_part_bbox_cord = np.stack(np.where(scaled_part_bbox), axis=1) + dist
        part_voxel_grid = np.full((resolution, resolution, resolution), 0, dtype='uint8')
        for each_cord in scaled_centered_part_bbox_cord:
            part_voxel_grid[each_cord[0], each_cord[1], each_cord[2]] = 1
        part_voxel_grid_list.append(part_voxel_grid)

        # get transformation matrix information
        translation = np.min(scaled_centered_part_bbox_cord, axis=0) - part_min_cord * scale_factor
        transformation_matrix = np.full((4, 4), 0, dtype='float32')
        transformation_matrix[0, 0] = transformation_matrix[1, 1] = transformation_matrix[2, 2] = scale_factor
        transformation_matrix[0, 3] = translation[0]
        transformation_matrix[1, 3] = translation[1]
        transformation_matrix[2, 3] = translation[2]
        transformation_matrix[3, 3] = 1.
        transformation_matrix_list.append(transformation_matrix)

    return np.asarray(part_voxel_grid_list), np.asarray(transformation_matrix_list), np.asarray(which_part)


def check_file_path(pcd_fp, binvox_fp):

    # check pcd has corresponding label and img
    pcd_dir = os.path.join(pcd_fp, 'points')
    pcd_label_dir = os.path.join(pcd_fp, 'points_label')
    pcd_img_dir = os.path.join(pcd_fp, 'seg_img')
    pcd_names = [os.path.splitext(path)[0] for path in os.listdir(pcd_dir)]
    pcd_names.sort()
    label_names = [os.path.splitext(path)[0] for path in os.listdir(pcd_label_dir)]
    label_names.sort()
    img_names = [os.path.splitext(path)[0] for path in os.listdir(pcd_img_dir)]
    img_names.sort()
    assert pcd_names == label_names == img_names

    # check every pcd has its corresponding binvox
    binvox_names = os.listdir(binvox_fp)
    for pcd_name in pcd_names:
        assert pcd_name in binvox_names


def process_data(resolution, pcd_fp, binvox_fp, output_fp, k=5):
    """
    :param pcd_fp: path of point cloud data. for example, './shapenetcore_partanno_segmentation_benchmark_v0/03001627/'
    :param binvox_fp: path of binvox files. for example, './ShapeNetVox32/03001627/'
    :param output_fp: path to save the processed data. for example, './datasets/'
    :param resolution: resolution to voxelize the point cloud data. should be the same as the resolution of binvox file
    :param k: knn parameter for the voxelization
    """

    # check all stuff
    redundancy = '.DS_Store'
    os.system(f'find {pcd_fp} -name "{redundancy}" -delete')
    os.system(f'find {binvox_fp} -name "{redundancy}" -delete')
    check_file_path(pcd_fp, binvox_fp)
    if not pcd_fp.endswith('/'):
        pcd_fp += '/'
    category_name = pcd_fp.split('/')[-2]
    output_fp = os.path.join(output_fp, category_name)

    shape_names = [os.path.splitext(each)[0] for each in os.listdir(os.path.join(pcd_fp, 'points'))]
    pb = Progbar(len(shape_names))
    for count, shape_name in enumerate(shape_names):
        shape_dir = os.path.join(output_fp, shape_name)
        if not os.path.exists(shape_dir):
            os.makedirs(shape_dir)

        binvox_path = os.path.join(binvox_fp, shape_name, 'model.binvox')
        with open(binvox_path, 'rb') as f:
            voxel_grid = binvox_rw.read_as_3d_array(f).data
        scipy.io.savemat(os.path.join(shape_dir, 'object_unlabeled.mat'), {'data': voxel_grid})
        visualization.save_visualized_img(voxel_grid, save_dir=os.path.join(shape_dir, 'object_unlabeled.png'))

        ref_label = get_reference_label(os.path.join(pcd_fp, 'points', f'{shape_name}.pts'),
                                        os.path.join(pcd_fp, 'points_label', f'{shape_name}.seg'),
                                        resolution)
        sur_label = get_surface_label(voxel_grid, ref_label)
        voxel_grid_label = get_voxel_grid_label(voxel_grid, sur_label, k)
        scipy.io.savemat(os.path.join(shape_dir, 'object_labeled.mat'), {'data': voxel_grid_label})
        visualization.save_visualized_img(voxel_grid_label, save_dir=os.path.join(shape_dir, 'object_labeled.png'))

        part_voxel_grid, transformation, which_part = get_seperated_part_and_transformation(voxel_grid_label)
        for part, trans, which in zip(part_voxel_grid, transformation, which_part):
            scipy.io.savemat(os.path.join(shape_dir, f'part{which}.mat'), {'data': part})
            scipy.io.savemat(os.path.join(shape_dir, f'part{which}_trans_matrix.mat'), {'data': trans})
            visualization.save_visualized_img(part, save_dir=os.path.join(shape_dir, f'part{which}.png'))
        pb.update(count+1)

    os.system(f'find {output_fp} -name "{redundancy}" -delete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is used to convert point cloud and its label to voxel '
                                                 'grid and its label.')

    # positional arguments
    parser.add_argument('resolution', help='resolution to voxelize the point cloud data. should be the same as the resolution of binvox file.')
    parser.add_argument('semantic_label_fp', help='path of the semantic label. For example, \'./shapenetcore_partanno_segmentation_benchmark_v0/03001627/\'')
    parser.add_argument('binvox_fp', help='path of binvox files. For example, \'./ShapeNetVox32/03001627/\'')
    parser.add_argument('output_fp', help='path to save the processed data. For example, \'./datasets/\'')

    # optional arguments
    parser.add_argument('-k', default=5, help='knn parameter for the voxelization. default 5')

    args = parser.parse_args()

    process_data(int(args.resolution), args.semantic_label_fp, args.binvox_fp, args.output_fp, int(args.k))
