# linear regression
# y = 8.92x1 - 15.63x2 + 9.75

import torch

x = []
y = []

for i in range(1, 21):
    x1 = i
    x2 = i+40

    x.append([x1, x2])

    x1 = (8.92*i) - (15.63*x2) + 9.75
    y.append([x1])

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, output_dim)

    def forward(self, x):
        r = self.l1(x)
        r1 = self.l2(r)

        return self.l3(r1)

    def fit(self, loss, optimizer, epochs, x, y):
        for epoch in range(epochs):
            yp = self.forward(x)
            los = loss(y, yp)
            los.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch+1} \t loss : {los}")


model = Model(x.shape[1], y.shape[1])

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.015)

# since .fit() is not inbuilt you have to create your own 
model.fit(loss, optimizer, 10000, x, y)

i = 43
yt = (8.92*i) - (15.63*(i+40)) + 9.75
xt = torch.tensor([i, i+40], dtype=torch.float32)
yp = model(xt)
print("Y test : ", yt)
print("Y pred : ", yp[0].item())
