# Reshape tensor

import torch

t = torch.randint(1, 9, (4, 4))
print(t)

t = t.view(2, 8)
print(t)

t = t.view(16)
print(t)

t = t.view(-1, 8)
print(t)

t = t.view(4, -1)
print(t) 
