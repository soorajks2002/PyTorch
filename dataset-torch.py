import torch
from torch.utils.data import Dataset
import numpy as np


class SampleDataset (Dataset):
    def __init__(self):
        xy = np.loadtxt("dataset.csv", delimiter=',',
                        skiprows=1, dtype=np.float32)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.n_samples


data = SampleDataset()
print(len(data))
print(data[56])



