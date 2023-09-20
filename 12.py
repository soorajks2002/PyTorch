import torch

x = torch.tensor([1, 2, 3, 4, 5, 6, 7])
y = torch.tensor([2, 4, 6, 8, 10, 12, 14])

w = torch.tensor(0.0, requires_grad=True)


def forward(x):
    return x*w


def loss(y, yp):
    return ((y-yp)**2).mean()


def gradient(los):
    los.backward()


print("f(96) before training : ", forward(96))

learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # forward pass
    yp = forward(x)

    # loss calculation
    l = loss(y, yp)

    # gradient calculation
    gradient(l)
    dw = w.grad

    # Weight updation
    with torch.no_grad():
        w -= (dw*learning_rate)

    # w.data -= (dw*learning_rate)

    # with torch.no_grad(): neglects the gradients in a variable
    # or else use variable.data to access the data of a variable whose gradients are also tracked

    with torch.no_grad():
        print(f"Epoch : {epoch+1} \t Weight : {w.data} \t Loss : {l}")

    # print(f"Epoch : {epoch+1} \t Weight : {w.data} \t Loss : {l}")

    w.grad.zero_()

print(f"f(96) after training for {epochs} epochs : ", forward(96).data.item())
