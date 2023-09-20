import torch
from torch.utils.data import Dataset, DataLoader
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


dataset = SampleDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

data_iter = iter(dataloader)

data = next(data_iter)
x, y = data

print(y)
