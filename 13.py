# design models -> input, output, forward
# construct models -> loss, optimizer
# training loop ->
#  forward pass -- make prediction
#  backward pass -- calculate gradients & weight updation

import torch

x = []
y = []

for i in range(1, 20):
    x.append([i])
    y.append([4*i])

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
xt = torch.tensor([67], dtype=torch.float32)

model = torch.nn.Linear(in_features=(1), out_features=(1))

print("Prediction before training : ", model(xt)[0].item())

learning_rate = 0.015
epochs = 5000

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    yp = model(x)
    l = loss(y, yp)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        [w, b] = model.parameters()
        print("weight value : ", w[0][0].item())

print("Prediction after training : ", model(xt)[0].item())
