# slicing operation on tensor

import torch

t = torch.randint(low=1, high=9, size=(2, 3))

print(t)
print(t[1])
print(t[1][2])

print(t[1, 2])
print(t[:, 0])
