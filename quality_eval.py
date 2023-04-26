import tensorflow as tf
import scipy.io
import importlib
import numpy as np
import os
import sys
PROJ_ROOT = os.path.abspath(__file__)[:-15]
sys.path.append(PROJ_ROOT)
import argparse
from utils import visualization
from utils.pick import get_pred_label


CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}


def evaluate_model(model_path,
                   mode='batch',
                   category='chair',
                   num_to_visualize=4,
                   single_shape_path=None,
                   H=32,
                   W=32,
                   D=32,
                   C=1,
                   visualize_decoded_part=False,
                   decoded_part_threshold=0.5,
                   transformed_part_threshold=0.5):

    # check category
    if category not in ['chair', 'table', 'airplane', 'lamp', 'guitar']:
        raise ValueError('category should be one of chair, table, airplane, guitar and lamp!')

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # check parameters
    if mode == 'batch':
        pass
    elif mode == 'single':
        if not single_shape_path.endswith('/'):
            single_shape_path += '/'
        category_code = single_shape_path.split('/')[-3]
        category = list(CATEGORY_MAP.keys())[list(CATEGORY_MAP.values()).index(category_code)]
    else:
        raise ValueError('mode should be one of batch and single!')

    # load weights and warm up model
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")
    hparam = importlib.import_module(f"results.{model_path.split('/')[-3]}.hparam")
    my_model = model.Model(hparam.hparam['max_num_parts'], hparam.hparam['bce_weight'], hparam.hparam['bce_weight_shape'],
                           3, hparam.hparam['use_attention'], hparam.hparam['keep_channel'], hparam.hparam['use_ac_loss'],
                           hparam.hparam['which_layer'], hparam.hparam['num_blocks'], hparam.hparam['num_heads'],
                           hparam.hparam['d_model'])
    my_model(warm_up_data)
    my_model.load_weights(model_path, by_name=True)

    # evaluation for every mode
    if mode == 'batch':
        dataset_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])
        all_shapes = os.listdir(dataset_path)
        idx = np.random.choice(len(all_shapes), num_to_visualize, replace=False)
        shapes_to_visualize = [all_shapes[each] for each in idx]

        for shape_code in shapes_to_visualize:
            # visualize ground truth label
            gt_label_path = os.path.join(dataset_path, shape_code, 'object_labeled.mat')
            gt_label = scipy.io.loadmat(gt_label_path)['data']
            visualization.visualize(gt_label, title=shape_code)

            # visualize predicted label
            shape_path = os.path.dirname(gt_label_path)
            gt_shape_path = os.path.join(shape_path, 'object_unlabeled.mat')
            visualize_pred_label(my_model, gt_shape_path, shape_code, visualize_decoded_part,
                                 decoded_part_threshold, transformed_part_threshold)

    elif mode == 'single':
        # visualize ground truth label
        shape_code = single_shape_path.split('/')[-2]
        gt_label = scipy.io.loadmat(os.path.join(single_shape_path, 'object_labeled.mat'))['data']
        visualization.visualize(gt_label, title=shape_code)

        # visualize predicted label
        gt_shape_path = os.path.join(single_shape_path, 'object_unlabeled.mat')
        visualize_pred_label(my_model, gt_shape_path, shape_code, visualize_decoded_part,
                             decoded_part_threshold, transformed_part_threshold)


def visualize_pred_label(model,
                         shape_path,
                         shape_code,
                         visualize_decoded_part=False,
                         decoded_part_threshold=0.5,
                         transformed_part_threshold=0.5):
    """
    :param model: tensorflow model
    :param shape_path: it should be a list when mode is 'exchange' and 'assembly', should be a str for other modes
    :param shape_code: it should be a list when mode is 'exchange' and 'assembly', should be a str for other modes
    :param visualize_decoded_part: whether to visualize decoded parts. only valid for 'batch' and 'single' mode
    :param decoded_part_threshold: threshold for decoded parts to be visualized
    :param transformed_part_threshold: threshold for transformed parts to be visualized
    """

    gt_shape = tf.convert_to_tensor(scipy.io.loadmat(shape_path)['data'], dtype=tf.float32)
    gt_shape = tf.expand_dims(gt_shape, 0)
    gt_shape = tf.expand_dims(gt_shape, 4)
    model_output = model(gt_shape, training=False)
    if visualize_decoded_part:
        pred = tf.squeeze(tf.where(model.stacked_decoded_parts > decoded_part_threshold, 1., 0.))
        pred = pred.numpy().astype('uint8')
        for part in pred:
            visualization.visualize(part, title=shape_code)
    else:
        pred = tf.squeeze(tf.where(model_output > transformed_part_threshold, 1., 0.))
        pred_label = get_pred_label(pred)
        visualization.visualize(pred_label, title=shape_code)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='this is the script to visualize reconstructed shape')

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-m', '--mode', default='batch', help='visualize a batch of shapes or just a single shape. '
                                                              'Valid values are batch and single. Default is batch')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('-n', '--num_to_visualize', default=4, help='the number of shape to be visualized. Only valid'
                                                                    'when \'mode\' is \'batch\'')
    parser.add_argument('-s', '--single_shape_path', default=None, help='path of the shape to be visualized. e.g. '
                                                          'datasets/03001627/1a6f615e8b1b5ae4dbbc9440457e303e. Only '
                                                          'valid when \'mode\' is \'single\'')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-v', '--visualize_decoded_part', action="store_true", help='whether to visualize decoded parts')
    parser.add_argument('-d', '--decoded_part_threshold', default=0.5, help='threshold of decoded parts to be visualized. '
                                                                            'Default is 0.5')
    parser.add_argument('-t', '--transformed_part_threshold', default=0.5, help='threshold of transformed parts to be visualized,'
                                                                                'Default is 0.5')
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path,
                   mode=args.mode,
                   category=args.category,
                   num_to_visualize=int(args.num_to_visualize),
                   single_shape_path=args.single_shape_path,
                   H=int(args.H),
                   W=int(args.W),
                   D=int(args.D),
                   C=int(args.C),
                   visualize_decoded_part=args.visualize_decoded_part,
                   decoded_part_threshold=float(args.decoded_part_threshold),
                   transformed_part_threshold=float(args.transformed_part_threshold))
