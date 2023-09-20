# backpropagation

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward pass
y_hat = w*x
loss = pow(y_hat-y, 2)

# calculate independent gradients and backward pass
loss.backward()

print(w.grad)
