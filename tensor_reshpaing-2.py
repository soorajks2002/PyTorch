from sklearn import datasets
import torch

x, y = datasets.make_regression(n_samples=10, n_features=2)

y = torch.tensor(y, dtype=torch.float32)
y = y.view(y.shape[0], 1)

print(y.shape)
print(y)
