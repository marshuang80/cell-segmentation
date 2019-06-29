from torch.utils.data import Dataset
#from scipy.ndimage.interpolation import rotate
from scipy import ndimage
import skimage
import imageio
import numpy as np
import h5py
import os
import random

class NucleiDataset(Dataset):

    def __init__(self, data_path, phase='train', transform=False):
        """Custom PyTorch Dataset for nuclei dataset

        Parameters
        ----------
            data_path: str
                path to the nuclei dataset hdf5 file
            phase: str, optional
                phase this dataset is used for (train, val. test)
        """

        self.data_path = data_path
        self.phase = phase
        self.transform = transform

        with h5py.File(self.data_path,"r") as h:
            self.data_names = list(h.keys())

            self.dim = 1    # decision to only use one channel (rgb are the same, a is all 1s)


    def __len__(self):

        return len(self.data_names)


    def __getitem__(self, idx):

        with h5py.File(self.data_path,"r") as h:
            data = h[self.data_names[idx]][:]

        x = data[0]
        x = np.expand_dims(x, axis=0)
        y = data[4]

        masks = data[4:]
       
        if self.transform:
            
            y = np.expand_dims(y, axis=0).astype(int)
            x, y = self.augmentation(x, y)

            #transform_x = self.transform.deepcopy()
            #transform_y = self.transform.deepcopy()

            #transform_x = transform_x.localize_random_state()

            #transform_x = transform_x.to_deterministic()
            #transform_y = transform_y.to_deterministic()

            #transform_y = transform_y.copy_random_state(transform_x)

            #x = transform_x.augment_image(x).copy()
            #y = transform_y.augment_image(y).copy()

            #transform = self.transform.to_deterministic()
            #transform = transform.localize_random_state()
            #x_t, y_t = self.transform(image=x, segmentation_maps=y)
            #x = x_t.copy()
            #y = y_t.copy()
        
        if self.phase == "val":
            return x, y, masks

        return x, y


    def augmentation(self, x, y):
        """augment input and mask
        """

        # vertical flip
        if random.random() < 0.5:
            x = np.flip(x, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()
        
        # horizontal flip
        if random.random() < 0.5:
            x = np.flip(x, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()

        # random rotation
        if random.random() < 0.5:
            angle = random.randint(-30, 30)
            x = ndimage.rotate(x, angle, (-2, -1), reshape=False)
            y = ndimage.rotate(y, angle, (-2, -1), reshape=False)

        # random cropping
        if random.random() < 0.5:
            proportion = random.uniform(0.7,1.0)
            
            x, y = self._random_crop(x, y, proportion)
        
        x = x.astype(np.double)
        y = y.astype(np.double)

        return x,y


    def _random_crop(self, x, y, proportion):

        ori_height = x.shape[-2]
        ori_width = x.shape[-1]
        ori_dim = x.shape[0]

        crop_height = int(ori_height * proportion)
        crop_width = int(ori_width * proportion)

        hori_start = random.randint(0, ori_width - crop_width -1)
        verti_start = random.randint(0, ori_height - crop_height -1)

        hori_end = hori_start + crop_width
        verti_end = verti_start + crop_height
        
        x = x[:,verti_start:verti_end, hori_start:hori_end]
        y = y[:,verti_start:verti_end, hori_start:hori_end]

        x = skimage.transform.resize(x, (ori_dim, ori_height, ori_width))
        y = skimage.transform.resize(y, (ori_dim, ori_height, ori_width), preserve_range=True)

        return x, y 
