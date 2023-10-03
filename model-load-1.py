import torch
from sklearn.datasets import make_regression


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

model_load_path = "model-save-1.pt"
model = torch.load(model_load_path)

xt, yt = make_regression(n_samples=10, n_features=5, n_targets=1)

xt = torch.tensor(xt, dtype=torch.float32)

print(model(xt))
