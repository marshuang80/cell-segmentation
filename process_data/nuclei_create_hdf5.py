import numpy as np
import h5py
import argparse
import imageio
import tqdm
import os
from glob import glob


def main(args):
    """Main function to parse in Nuclei Dataset from Kaggle and store as HDF5

    Parameters
    ----------
        args: ArgumentParser()
            input_dir: str
                directory of the Nuclei data
            output_dir: str
                path to the HDF5 output directory
    """

    # create hdf5 
    hdf5_fn = h5py.File(os.path.join(args.output_dir, "data_360.hdf5"), "a")

    # get all data directory
    data_dirs = glob(os.path.join(args.input_dir, "*/"))

    with tqdm.tqdm(total=len(data_dirs), unit="folder") as progress_bar:
        for path in data_dirs:

            data_name = path.split("/")[-2]
            x, y, masks = parse_data(path)

            # TODO only use majority size for now
            if x is None:
                progress_bar.update(1)
                continue

            # stack x and y together
            y = np.expand_dims(y, axis=0)
            data = np.vstack((x,y,masks))

            hdf5_fn.create_dataset(str(data_name), data=data, dtype=np.float, chunks=True)
            progress_bar.update(1)
    hdf5_fn.close()


def parse_data(path):

    # define data folders
    x_path = os.path.join(path, "images/")
    y_path = os.path.join(path, "masks/")

    # get all data paths 
    x_file = glob(os.path.join(x_path, "*.png"))[0]
    y_files = glob(os.path.join(y_path, "*.png"))

    # parse in data
    x = imageio.imread(x_file)

    # TODO only using majority shape
    if x.shape != (256, 256, 4):
        return None, None, None

    masks = np.array([imageio.imread(y) for y in y_files])
    y = np.zeros_like(masks[0])
    for y_raw in masks:
        y = np.maximum(y, y_raw)

    # normalize
    x = x / 255.0
    y = y / 255.0
    masks = masks / 255.0

    # fix dimentions
    x = np.transpose(x, (2,0,1))    # channels first

    return x, y, masks


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    main(args)
