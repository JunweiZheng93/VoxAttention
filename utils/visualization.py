from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import argparse
import scipy.io


def visualize(x, max_num_parts=8, show_axis=False, show_grid=False, cmap='Set2', title=None):
    """
    :param x: ndarray. input data to be visualized.
    :param max_num_parts: maximal number of parts of the category. e.g. the maximal number of parts for the category
           chair is 4.
    :param show_axis: show the axis of the figure or not.
    :param show_grid: show the grid of the figure or not.
    :param cmap: name of the cmap.
    :param title: name of the window.
    """

    x = np.transpose(x, (0, 2, 1))
    fig = plt.figure(title)
    ax = fig.add_subplot(projection='3d')
    new_cmap = get_cmap(max_num_parts, cmap)
    label_color = np.take(new_cmap, x, axis=0)
    ax.voxels(x, facecolors=label_color)
    if not show_axis:
        plt.axis('off')
    if not show_grid:
        plt.grid('off')
    plt.show()
    plt.close(fig)


def get_cmap(num_points, cmap):
    selected_cmap = cm.get_cmap(cmap, num_points)
    if not isinstance(selected_cmap, ListedColormap):
        raise ValueError(f'cmap should be <class \'matplotlib.colors.ListedColormap\'>, but got {type(selected_cmap)}')
    new_cmap = np.ones((num_points + 1, 4))
    new_cmap[1:] = selected_cmap.colors
    return new_cmap


def save_visualized_img(x, save_dir, max_num_parts=8, cmap='Set2', show_axis=False, show_grid=False, is_unlabeled_gt=False):
    """
    :param x: ndarray. input data to be visualized.
    :param save_dir: path to save the visualized image.
    :param max_num_parts: maximal number of parts of the category. e.g. the maximal number of parts for the category
           chair is 4.
    :param show_axis: show the axis of the figure or not.
    :param show_grid: show the grid of the figure or not.
    :param cmap: name of the cmap.
    :return:
    """
    x = np.transpose(x, (0, 2, 1))
    fig = plt.figure(figsize=(3.2, 2.4))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    new_cmap = get_cmap(max_num_parts, cmap)
    label_color = np.take(new_cmap, x, axis=0)
    if is_unlabeled_gt:
        ax.voxels(x, facecolors='lightgray')
    else:
        ax.voxels(x, facecolors=label_color)
    if not show_axis:
        plt.axis('off')
    if not show_grid:
        plt.grid('off')
    plt.savefig(save_dir)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the script for visualizing .mat files')

    parser.add_argument('mat_fp', help='file path of the mat file')
    parser.add_argument('--show_axis', action='store_true', help='show coordinate axis')
    parser.add_argument('--show_grid', action='store_true', help='show coordinate grid')
    parser.add_argument('--cmap', default='Set2', help='color map for the visualization. default Set2')

    args = parser.parse_args()

    data = scipy.io.loadmat(args.mat_fp)
    x = data['data']
    visualize(x, show_axis=args.show_axis, show_grid=args.show_grid, cmap=args.cmap)
