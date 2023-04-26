import os
import sys
PROJ_ROOT = os.path.abspath(__file__)[:-23]
sys.path.append(PROJ_ROOT)
import platform
import argparse


def obj_to_binvox(resolution, obj_category_path, output_path):
    if platform.system() == 'Darwin':
        redundancy = '.DS_Store'
        os.system(f'find {obj_category_path} -name "{redundancy}" -delete')
    os.system(f'chmod 755 {os.path.join(PROJ_ROOT, "utils", "binvox")}')
    all_shapes = os.listdir(obj_category_path)
    for shape in all_shapes:
        output_shape_path = os.path.join(output_path, shape)
        if not os.path.exists(output_shape_path):
            os.makedirs(output_shape_path)
        obj_path = os.path.join(obj_category_path, shape, 'model.obj')
        os.system(f'{os.path.join(PROJ_ROOT, "utils", "binvox")} -e -cb -d {resolution} {obj_path}')
        os.system(f'mv {os.path.join(obj_category_path, shape, "model.binvox")} {os.path.join(output_shape_path, "model.binvox")}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('resolution', help='resolution of the binvox, for example, 32')
    parser.add_argument('obj_category_path', help='category path containing obj file, for example, \'~/Desktop/03467517\' ')
    parser.add_argument('output_path', help='output path of the binvox files')

    args = parser.parse_args()

    obj_to_binvox(int(args.resolution),
                  args.obj_category_path,
                  args.output_path)
