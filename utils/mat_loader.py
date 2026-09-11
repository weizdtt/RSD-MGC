import torch
import numpy as np
import scipy.io as scio


class Cub:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = scio.loadmat(data_path)

    def get_data(self):
        dataset = [torch.from_numpy(normalize(self.raw_data['X'][0][0]).astype('float32')),
                   torch.from_numpy(normalize(self.raw_data['X'][0][1]).astype('float32')),
                   torch.from_numpy(self.raw_data['gt']).squeeze(1)]
        return dataset


def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile(
        (np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x
