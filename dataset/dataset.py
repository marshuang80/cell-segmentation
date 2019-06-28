from torch.utils.data import Dataset
import numpy as np
import h5py
import os

class NucleiDataset(Dataset):

    def __init__(self, data_path, phase='train'):
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
        
        if self.phase == "train":
            return x, y
        else:
            return x, y, masks
