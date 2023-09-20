import torch
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=5000, n_features=1)
y = y.reshape(-1, 1)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
# y = y.view(y.shape[0], 1)


class Model (torch.nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(Model, self).__init__()
        self.l = torch.nn.Linear(input_feature_size, output_feature_size)

    def forward(self, x):
        return self.l(x)

    def fit(self, loss, optimizer, epochs, batch_size, x, y):
        for epoch in range(epochs):
            print("\n")
            for i, batch_start in enumerate(range(0, x.shape[0], batch_size)):
                print(f"Epoch : {epoch+1} \t Batch : {i+1}")
                xt = x[batch_start:batch_start+batch_size]
                yt = y[batch_start:batch_start+batch_size]
                yp = self.forward(xt)
                l = loss(yt, yp)
                l.backward()
                optimizer.step()
                optimizer.zero_grad()


model = Model(x.shape[1], y.shape[1])
loss = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)

model.fit(loss, optimizer, 10, 1000, x, y)

with torch.no_grad():
    # print(model(x))
    # print(y)
    pass
