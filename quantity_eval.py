import tensorflow as tf
import importlib
import os
import sys
PROJ_ROOT = os.path.abspath(__file__)[:-16]
sys.path.append(PROJ_ROOT)
import argparse
from utils import dataloader
from utils.pick import configure_gpu
from tensorflow.keras.utils import Progbar


def evaluate_model(model_path,
                   which_gpu=0,
                   H=32,
                   W=32,
                   D=32,
                   C=1):

    configure_gpu(which_gpu)

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # load weights and warm up model
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{model_path.split('/')[-3]}.hparam")
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], hparam.hparam['bce_weight_shape'],
                           hparam.hparam['training_process'], hparam.hparam['use_attention'], hparam.hparam['keep_channel'],
                           hparam.hparam['use_ac_loss'], hparam.hparam['which_layer'], hparam.hparam['num_blocks'],
                           hparam.hparam['num_heads'], hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    # get dataset
    training_set, test_set = dataloader.get_dataset(category=hparam.hparam['category'], batch_size=hparam.hparam['batch_size'],
                                                    split_ratio=hparam.hparam['split_ratio'], max_num_parts=hparam.hparam['max_num_parts'])

    transformation_mse_tracker = tf.keras.metrics.Mean()
    part_mIoU_tracker_list = [tf.keras.metrics.MeanIoU(2) for i in range(my_model.num_parts)]
    all_part_mIoU_tracker = tf.keras.metrics.MeanIoU(2)
    shape_mIoU_tracker = tf.keras.metrics.MeanIoU(2)
    part_symmetry_score_tracker_list = [tf.keras.metrics.Mean() for i in range(my_model.num_parts)]
    gt_part_symmetry_score_tracker_list = [tf.keras.metrics.Mean() for i in range(my_model.num_parts)]
    shape_symmetry_score_tracker = tf.keras.metrics.Mean()
    gt_shape_symmetry_score_tracker = tf.keras.metrics.Mean()

    pb = Progbar(len(test_set))
    print('Your model is being evaluated, please wait...')

    if hparam.hparam['training_process'] == 1 or hparam.hparam['training_process'] == '1':
        for x, labels, trans in test_set:
            parts = my_model(x, training=False)
            parts = tf.transpose(tf.where(parts > 0.5, 1., 0.), (1, 0, 2, 3, 4, 5))
            labels = tf.transpose(labels, (1, 0, 2, 3, 4, 5))
            for gt, part, part_mIoU_tracker, part_symmetry_score_tracker, gt_part_symmetry_score_tracker in \
                    zip(labels, parts, part_mIoU_tracker_list, part_symmetry_score_tracker_list, gt_part_symmetry_score_tracker_list):
                part_mIoU_tracker.update_state(gt, part)
                all_part_mIoU_tracker.update_state(gt, part)
                gt_part_symmetry_score_tracker.update_state(cal_symmetry_score(gt))
                part_symmetry_score_tracker.update_state(cal_symmetry_score(part))
        for i in range(my_model.num_parts):
            print(f'Part{i+1}_mIoU: {part_mIoU_tracker_list[i].result()}')
        print(f'Part_mIoU: {all_part_mIoU_tracker.result()}')
        for i in range(my_model.num_parts):
            print(f'GT_Part{i+1}_Symmetry_Score: {gt_part_symmetry_score_tracker_list[i].result()}')
        for i in range(my_model.num_parts):
            print(f'Part{i+1}_Symmetry_Score: {part_symmetry_score_tracker_list[i].result()}')

    elif hparam.hparam['training_process'] == 2 or hparam.hparam['training_process'] == '2':
        for i, (x, labels, trans) in enumerate(test_set):
            theta = my_model(x, training=False)
            trans_mse = my_model._cal_transformation_loss(trans, theta) * 2 / my_model.num_parts
            transformation_mse_tracker.update_state(trans_mse)
            shapes = model.Resampling()((my_model.stacked_decoded_parts, theta))
            shapes = tf.where(tf.reduce_max(shapes, axis=1) > 0.5, 1., 0.)
            shape_mIoU_tracker.update_state(x, shapes)
            gt_shape_symmetry_score_tracker.update_state(cal_symmetry_score(x))
            shape_symmetry_score_tracker.update_state(cal_symmetry_score(shapes))
            pb.update(i+1)
        print(f'Transformation_MSE: {transformation_mse_tracker.result()}')
        print(f'Shape_mIoU: {shape_mIoU_tracker.result()}')
        print(f'GT_Shape_Symmetry_Score: {gt_shape_symmetry_score_tracker.result()}')
        print(f'Shape_Symmetry_Score: {shape_symmetry_score_tracker.result()}')

    else:
        for i, (x, labels, trans) in enumerate(test_set):
            shapes = my_model(x, training=False)
            trans_mse = my_model._cal_transformation_loss(trans, my_model.theta) * 2 / my_model.num_parts
            transformation_mse_tracker.update_state(trans_mse)
            shapes = tf.where(tf.reduce_max(shapes, axis=1) > 0.5, 1., 0.)
            parts = tf.transpose(tf.where(my_model.stacked_decoded_parts > 0.5, 1., 0.), (1, 0, 2, 3, 4, 5))
            labels = tf.transpose(labels, (1, 0, 2, 3, 4, 5))
            for gt, part, part_mIoU_tracker, part_symmetry_score_tracker, gt_part_symmetry_score_tracker in \
                    zip(labels, parts, part_mIoU_tracker_list, part_symmetry_score_tracker_list, gt_part_symmetry_score_tracker_list):
                part_mIoU_tracker.update_state(gt, part)
                all_part_mIoU_tracker.update_state(gt, part)
                gt_part_symmetry_score_tracker.update_state(cal_symmetry_score(gt))
                part_symmetry_score_tracker.update_state(cal_symmetry_score(part))
            shape_mIoU_tracker.update_state(x, shapes)
            gt_shape_symmetry_score_tracker.update_state(cal_symmetry_score(x))
            shape_symmetry_score_tracker.update_state(cal_symmetry_score(shapes))
            pb.update(i+1)
        for i in range(my_model.num_parts):
            print(f'Part{i+1}_mIoU: {part_mIoU_tracker_list[i].result()}')
        print(f'Part_mIoU: {all_part_mIoU_tracker.result()}')
        for i in range(my_model.num_parts):
            print(f'GT_Part{i+1}_Symmetry_Score: {gt_part_symmetry_score_tracker_list[i].result()}')
        for i in range(my_model.num_parts):
            print(f'Part{i + 1}_Symmetry_Score: {part_symmetry_score_tracker_list[i].result()}')
        print(f'Transformation_MSE: {transformation_mse_tracker.result()}')
        print(f'Shape_mIoU: {shape_mIoU_tracker.result()}')
        print(f'GT_Shape_Symmetry_Score: {gt_shape_symmetry_score_tracker.result()}')
        print(f'Shape_Symmetry_Score: {shape_symmetry_score_tracker.result()}')


def cal_symmetry_score(inputs):
    reflections = flip(inputs)
    matched_voxel = tf.cast(tf.logical_and(tf.cast(inputs, dtype=tf.bool), tf.cast(reflections, dtype=tf.bool)), dtype=tf.float32)
    symmetry_score = tf.reduce_sum(matched_voxel) / tf.reduce_sum(inputs)
    return symmetry_score


def flip(input_fmap):

    B = input_fmap.shape[0]
    H = input_fmap.shape[1]
    W = input_fmap.shape[2]
    D = input_fmap.shape[3]

    x = tf.cast(tf.linspace(0, W - 1, W), dtype=tf.uint8)
    y = tf.cast(tf.linspace(0, H - 1, H), dtype=tf.uint8)
    z = tf.cast(tf.linspace(0, D - 1, D), dtype=tf.uint8)
    x_t, y_t, z_t = tf.meshgrid(x, y, z)

    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])
    z_t_flat = tf.reshape(z_t, [-1])

    ones = tf.ones_like(x_t_flat, dtype=tf.uint8)
    sampling_grid = tf.stack([y_t_flat, x_t_flat, z_t_flat, ones])
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, [B, 1, 1])
    sampling_grid = tf.cast(sampling_grid, 'float32')

    theta = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, D-1]], dtype='float32')
    theta = tf.expand_dims(theta, axis=0)
    theta = tf.tile(theta, [B, 1, 1])

    batch_grids = theta @ sampling_grid
    batch_grids = tf.reshape(batch_grids, [B, 3, H, W, D])
    batch_grids = tf.cast(batch_grids, dtype='int32')

    y_s = batch_grids[:, 0, :, :, :]
    x_s = batch_grids[:, 1, :, :, :]
    z_s = batch_grids[:, 2, :, :, :]

    indices = tf.stack([y_s, x_s, z_s], axis=4)
    return tf.gather_nd(input_fmap, indices, batch_dims=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-gpu', default=0, help='use which gpu. Default is 0')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path,
                   which_gpu=int(args.gpu),
                   H=int(args.H),
                   W=int(args.W),
                   D=int(args.D),
                   C=int(args.C))
