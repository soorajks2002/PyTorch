import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)

x, xt, y, yt = train_test_split(x, y, test_size=0.2)

sc = StandardScaler()
x = sc.fit_transform(x)
xt = sc.transform(xt)

x = torch.tensor(x, dtype=torch.float32)
xt = torch.tensor(xt, dtype=torch.float32)

y = torch.tensor(y, dtype=torch.float32)
yt = torch.tensor(yt, dtype=torch.float32)


class Model(torch.nn.Module):
    def __init__(self, input_feature_size):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_feature_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = Model(x.shape[1])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 10

for epoch in range(epochs):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} \t Loss : {loss.item()}")
