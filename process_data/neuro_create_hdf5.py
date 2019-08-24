import glob
import json
import imageio
import h5py
import numpy as np
import os
import argparse


def main(args):
    files = glob.glob(os.path.join(args.input_dir, "*"))
    file_names = [file.split("_")[1].split(".")[0] for file in files if "image" in file]

    #for name in file_names:
    for name in file_names:
        hdf5 = h5py.File((os.path.join(args.output_dir, "data.hdf5"), "a")

        file_path = f"./train/image_{name}.tif"
        mask_path = f"./train/mask_{name}.tif"
        regions_path = f"./train-regions/regions_{name}.json"
        
        img = imageio.imread(file_path)
        # might need average
        img = img[:,:,0]
        mask = imageio.imread(mask_path)
        mask[mask > 0] = 1

        regions = json.load(open(regions_path))
        volume = [img, img, img, mask]
        
        for region in regions['regions']:
            coords = np.array(region['coordinates'])

            instance = np.zeros_like(img)
            instance[[coords[:,0],coords[:,1]]] = 1
            
            volume.append(instance)
            
        img_stack = np.stack(volume)
        
        hdf5_train.create_dataset(str(name), data=img_stack, dtype=np.int, chunks=True)

    hdf5.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    main(args)

