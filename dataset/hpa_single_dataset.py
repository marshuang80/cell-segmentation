from torch.utils.data import Dataset
from scipy import ndimage
from .augmentation import augmentation
import skimage
import imageio
import numpy as np
import h5py
import os
import random

class HPASingleDataset(Dataset):

    def __init__(self, data_path, phase='train', transform=False, percent_hole=0.4):
        """Custom PyTorch Dataset for hpa dataset

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
        self.percent_hole = percent_hole
        self.target_dim = 1


        with h5py.File(self.data_path,"r") as h:
            self.data_names = list(h.keys())

            self.dim = 1    # decision to only use one channel (rgb are the same, a is all 1s)


    def __len__(self):

        return len(self.data_names)


    def __getitem__(self, idx):

        with h5py.File(self.data_path,"r") as h:
            data = h[self.data_names[idx]][:]

        data = data / 255.

        size = len(list(data.reshape(-1)))

        rand_holes = np.ones(size)

        idx = np.random.choice(size, int(size*self.percent_hole), replace=False)
        rand_holes[idx] = 0 
        rand_holes = rand_holes.reshape(data.shape)
        
        y = data.copy()
        x = data.copy()
        
        x = x * rand_holes

        x = np.expand_dims(x, 0)
        if self.transform: 
            x,y = augmentation(x,y)


        return x, y
