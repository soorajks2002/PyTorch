import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SampleDataset (Dataset):
    def __init__(self):
        xy = np.loadtxt("dataset.csv", delimiter=',',
                        skiprows=1, dtype=np.float32)
        self.x = xy[:, 1:]
        self.y = xy[:, 0]
        self.length = xy.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.length


sample_dataset = SampleDataset()

dataloader = DataLoader(dataset=sample_dataset, batch_size=120, shuffle=False)

for i in dataloader:
    print(len(i[0]))
