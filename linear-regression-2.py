from sklearn import datasets
import torch

x, y = datasets.make_regression(n_samples=120, n_features=2)
t = []
for a in y:
    t.append([a])
y = t

xt = x[110:]
yt = y[110:]

x = x[:110]
y = y[:110]

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

xt = torch.tensor(xt, dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        # self.l1 = torch.nn.Linear(input_dim, 3)
        # self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # r = self.l1(x)
        # r1 = self.l2(r)

        return self.l3(x)

    def fit(self, loss, optimizer, epochs, x, y):
        for epoch in range(epochs):
            yp = self.forward(x)
            los = loss(y, yp)
            los.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch+1} \t loss : {los.item()}")


model = Model(x.shape[1], y.shape[1])

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# since .fit() is not inbuilt you have to create your own
model.fit(loss, optimizer, 10000, x, y)

print(yt)
print(model(xt))
