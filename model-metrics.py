import torch
import torchmetrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)

x, xt, y, yt = train_test_split(x, y, test_size=0.15)

scaler = StandardScaler()
x = scaler.fit_transform(x)
xt = scaler.transform(xt)

x = torch.tensor(x, dtype=torch.float32)
xt = torch.tensor(xt, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
yt = torch.tensor(yt, dtype=torch.float32)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


model = NeuralNetwork(x.shape[1])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)

epochs = 10

for epoch in range(epochs):
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"epoch : {epoch+1}/{epochs} \t loss : {loss.item():.4f}")

with torch.no_grad():
    model.eval()
    y_pred = model(xt)
    y_pred = [[1] if x[0] > 0.5 else [0] for x in y_pred]
    y_pred = torch.tensor(y_pred, dtype=torch.float32)

    metric = torchmetrics.classification.BinaryAccuracy()
    result = metric(y_pred, yt)

    print(f"Accuracy of the model on test data is {result.item()*100}")
