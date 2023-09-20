import torch

x = torch.randn(3, requires_grad=True)  # requires_grad is needed for gradients

print(x)

y = x + 10
print(y)

y = x*10
print(y)

y = x/10
print(y)

y = x-10
print(y)

y = x % 10
print(y)

y = x.mean()
print(y)

y = x.median()
print(y)

y = x.max()
print(y)

y = x + 10
print(y)
y = y.mean()
print(y)
print(x.grad)
y.backward()  # calculates dy/dx
print(x.grad)

# y.backward() will only work if y is a scalar value i.e. single value
# if y = [0, 1, 2, 3] something like this where y is of size 4 y.backward won't work as y is not scalar
# y should be a single value liek y=1.2 or y=558 etc

x = torch.randn(6, 6, requires_grad=True)
y = x+10
v = torch.randn(6, 6)

# if y value is not scalar then you should pass the jacobian vector seperatly to the y.backward() function
# the jacobian vector should be of same size as y

y.backward(v)
print(x.grad)
