
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class dataset(torch.utils.data.Dataset):
    def __init__(self):
        n_samples = 1000
        ratio = 0.1
        self.x, self.y = make_regression(
            n_samples=n_samples, n_features=5, n_targets=1, random_state=11)

        ss = train_test_split(self.x, self.y, test_size=ratio, shuffle=True)

        self.x = torch.tensor(ss[0], dtype=torch.float32)
        self.xt = torch.tensor(ss[1], dtype=torch.float32)
        self.y = torch.tensor(ss[2], dtype=torch.float32)
        self.yt = torch.tensor(ss[3], dtype=torch.float32)

        self.y = self.y.view(-1, 1)
        self.yt = self.yt.view(-1, 1)

        # print(self.x.shape, self.xt.shape, self.y.shape, self.yt.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ind):
        return (self.x[ind], self.y[ind])


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = torch.nn.Linear(5, 10)
        self.a1 = torch.nn.LeakyReLU()
        self.l2 = torch.nn.Linear(10, 3)
        self.a2 = torch.nn.LeakyReLU()
        self.l3 = torch.nn.Linear(3, 1)

    def __call__(self, x):
        out = self.l1(x)
        out = self.a1(out)
        out = self.a2(self.l2(out))
        out = self.l3(out)

        return out


data = dataset()
data_loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)

model = NeuralNetwork()

epochs = 10

optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

losses = []

for epoch in range(epochs):

    for x, y in data_loader:
        output = model(x)
        loss = criterion(output, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


model_save_path = "model-save-2.pt"
torch.save(model.state_dict(), model_save_path)
