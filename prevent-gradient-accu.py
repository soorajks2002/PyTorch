import torch

t = torch.ones(4, requires_grad=True)

for i in range(1):
    y = (t*3).mean()
    y.backward()
print(t.grad)


t = torch.ones(4, requires_grad=True)

# the gradient keeps on adding, hence the final value is accumplation of al the previous values
for i in range(3):
    y = (t*3).mean()
    y.backward()
print(t.grad)


# to prevent addition into previous values, we have to clear the gradient after operation

t = torch.ones(4, requires_grad=True)

for i in range(3):
    y = (t*3).mean()
    y.backward()
    t.grad.zero_()

print(t.grad)
