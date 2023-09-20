# Neuural Network using OOPS

import torch

x = []
y = []
for i in range(1, 10):
    x.append([i])
    y.append([(4*i)-7.29])

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


learning_rate = 0.012
epochs = 10000

model = LinearRegression(x.shape[1], y.shape[1])

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    yp = model(x)
    l = loss(y, yp)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

w, b = model.parameters()
print(w, b)
