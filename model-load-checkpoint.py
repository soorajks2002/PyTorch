import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


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


model = NeuralNetwork()
criterion = torch.nn.MSELoss()

model_load_path = "model-checkpoint-2.pt"
checkpoint = torch.load(model_load_path)
model.load_state_dict(checkpoint['model'])

xt, yt = make_regression(n_samples=1000, n_features=5,
                         n_targets=1, random_state=11)

_, xt, _, yt = train_test_split(xt, yt, test_size=0.1)

xt = torch.tensor(xt, dtype=torch.float32)
yt = torch.tensor(yt, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    output = model(xt)

loss = criterion(output, yt)
print(loss)
