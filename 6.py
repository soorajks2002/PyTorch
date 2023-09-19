import torch

x = torch.randn(3, requires_grad=True)
print(x)

x.requires_grad_(False)
print(x)

x = torch.randn(3, requires_grad=True)
print(x)

y = x.detach()
print(x)
print(y)

x = torch.randn(3, requires_grad=True)
print(x)

with torch.no_grad():
    y = x
    print(x)
    print(y)
    y = x + 2
    print(x)
    print(y)
