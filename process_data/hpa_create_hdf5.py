from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import glob

import h5py
import tqdm
import imageio
import numpy as np


def load_ids(input_dir):
    red_filenames = glob.glob(input_dir + '/*_red.png')
    green_filenames = glob.glob(input_dir + '/*_green.png')
    blue_filenames = glob.glob(input_dir + '/*_blue.png')
    yellow_filenames = glob.glob(input_dir + '/*_yellow.png')

    return [filename[:-8] for filename in red_filenames]


def process(id_lists, output_dir):
    
    hdf5_fn = h5py.File(os.path.join(output_dir, 'data.hdf5'), "a")

    for id_str in tqdm.tqdm(id_lists):

        output_filename = os.path.join(output_dir, os.path.basename(id_str) + '.png')
        if os.path.exists(output_filename):
            continue

        red_filename = id_str + '_red.png'
        green_filename = id_str + '_green.png'
        blue_filename = id_str + '_blue.png'
        yellow_filename = id_str + '_yellow.png'

        if not os.path.exists(red_filename) or \
           not os.path.exists(green_filename) or \
           not os.path.exists(blue_filename) or \
           not os.path.exists(yellow_filename):
            continue

        red = imageio.imread(red_filename)
        green = imageio.imread(green_filename)
        blue = imageio.imread(blue_filename)
        yellow = imageio.imread(yellow_filename)

        stacked = np.stack([red, green, blue, yellow], axis=2)

        imageio.imsave(output_filename, stacked)
        result = imageio.imread(output_filename)

        hdf5_fn.create_dataset(str(id_str), data=stacked, dtype=np.uint8, chunks=True)

    hdf5_fn.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', dest='input_dir',
                        help='the directory of the input images',
                        default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='the directory of the output images',
                        default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.input_dir is not None
    assert args.output_dir is not None

    os.makedirs(args.output_dir, exist_ok=True)

    id_lists = load_ids(args.input_dir)
    process(id_lists, args.output_dir)


if __name__ == '__main__':
    main()
