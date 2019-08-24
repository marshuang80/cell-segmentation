from torch.utils.data import Dataset
from scipy import ndimage
from .augmentation import augmentation
import skimage
import imageio
import numpy as np
import h5py
import os
import random

class NeuroDataset(Dataset):

    def __init__(self, data_path, phase='train', transform=False, target_channels="3"):
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
        
        if "," in target_channels: 
            self.target_channels = [int(c) for c in targat_channels.split(',')]
        else:
            self.target_channels = [int(target_channels)]

        self.target_dim = len(self.target_channels)

        with h5py.File(self.data_path,"r") as h:
            self.data_names = list(h.keys())

            self.dim = 1    


    def __len__(self):

        return len(self.data_names)


    def __getitem__(self, idx):

        with h5py.File(self.data_path,"r") as h:
            data = h[self.data_names[idx]][:]

        x = data[0]
        x = np.expand_dims(x, axis=0)
        y = data[self.target_channels]

        if self.transform:

            x, y = augmentation(x, y)

        return x, y
