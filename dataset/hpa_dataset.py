from torch.utils.data import Dataset
from scipy import ndimage
import skimage
import imageio
import numpy as np
import h5py
import os
import random

class HPADataset(Dataset):

    def __init__(self, data_path, phase='train', transform=False):
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

        with h5py.File(self.data_path,"r") as h:
            self.data_names = list(h['/raw']['train'].keys())

            self.dim = 1    # decision to only use one channel (rgb are the same, a is all 1s)


    def __len__(self):

        return len(self.data_names)


    def __getitem__(self, idx):

        with h5py.File(self.data_path,"r") as h:
            data = h['/raw']['train'][self.data_names[idx]][:]

        data = np.transpose(data, (2,0,1))
        data = data / 255.
        
        y = data[2]    # get blue nuclei channel
        y[y > 0.5] = 1.0    # try out different thresholds 
        y = np.expand_dims(y, 0)

        x = np.mean(data, axis=0)
        x = np.expand_dims(x, 0)
       
        if self.transform:
            
            y = np.expand_dims(y, axis=0).astype(int)
            x, y = self.augmentation(x, y)
        
        return x, y
