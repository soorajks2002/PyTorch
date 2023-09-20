import torch
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_regression


class SampleDataset(Dataset):
    def __init__(self, n_samples, n_features):
        x, y = make_regression(n_samples=n_samples, n_features=n_features)
        y = y.reshape(-1, 1)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = y.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.l = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.l(x)

    def fit(self, dataset, epochs, loss, optimizer, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        n_iterations = math.ceil(len(dataset)/batch_size)

        for epoch in range(epochs):
            print(f"Epoch : {epoch+1}")
            for iteration, data in enumerate(dataloader):
                print(f"batch {iteration+1} / {n_iterations}")

                x, y = data

                yp = self.forward(x)
                l = loss(y, yp)
                l.backward()
                optimizer.step()
                optimizer.zero_grad()


dataset = SampleDataset(n_samples=1000, n_features=3)

model = Model(len(dataset[0][0]), len(dataset[0][1]))

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

model.fit(dataset=dataset, epochs=1, loss=loss,
          optimizer=optimizer, batch_size=120)
