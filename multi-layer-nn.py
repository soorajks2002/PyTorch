import torch
from math import sqrt

x = []
y = []
for i in range(1, 16):
    x.append([i])
    y.append([(5.6*i)+23.87])

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, 8)
        self.l2 = torch.nn.Linear(8, output_dim)

    def forward(self, x):
        r1 = self.l1(x)
        r2 = self.l2(r1)
        return r2


model = Model(x.shape[1], y.shape[1])

loss = torch.nn.MSELoss()

learning_rate = 0.015
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

epochs = 2000

for i in range(epochs):
    yp = model(x)
    l = loss(y, yp)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

xt = torch.tensor([23], dtype=torch.float32)
yt = (5.6*23)+23.87

print("y test : ", yt)
print("prediction : ", model(xt)[0].item())
