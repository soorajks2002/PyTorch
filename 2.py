import torch

x = torch.rand(size=(2, 3))
y = torch.rand(size=(2, 3))

add = x+y
print(add)

add = torch.add(x, y)
print(add)

print(y.add(x))

# y.add(x) is not an inplace operation while y.add_(x) is an inplace operation
# any functino with _ at the end ia an inplace funcion in PyTorch

y.add_(x)
print(y)


# other operation function in pytorch
# .sub()
# .mul()
# .div()