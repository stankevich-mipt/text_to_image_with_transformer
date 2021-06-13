import os
import io
import h5py
import numpy as np
import torch
import torch.nn.functional as F


from PIL import Image
from torch.utils.data import Dataset

class VQVAEDataset(Dataset):

    def __init__(self, datasetFile, transform=None, split=0):
        
        self.datasetFile     = datasetFile
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = lambda x: int(np.array(x))

        self.dataset = h5py.File(self.datasetFile, mode='r')
        self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()]

    def __len__(self):

        return len(self.dataset_keys) 

    def __getitem__(self, idx):
        
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]

        image = bytes(np.array(example['img']))
        image = Image.open(io.BytesIO(image)).resize((64, 64))
        image = torch.FloatTensor(np.array(image)).div_(255.).permute(2, 0, 1)
        
        return image