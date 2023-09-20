# BASIC NEURAL NETWORK

import numpy as np

x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10, 12], dtype=np.float32)

w = 0.0

# Basic sekletel algorithm
# y_hat = w*x
# loss = (y_hat-y)**2
# loss.backward()
# w.grad
# w.grad.zero_()


def forward(x):
    return w*x


def loss(y, y_prediction):
    # loss function = MSE (Mean Squared Error)
    return ((y-y_prediction)**2).mean()


def gradient(x, y, y_prediction):
    # derivative of loss function i.e. MSE
    return np.dot(2*x, y_prediction-y).mean()


print("f(5) before training : ", forward(5))

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # forward propagation
    y_prediction = forward(x)

    # calculate loss function
    los = loss(y, y_prediction)

    # calculate gradient
    grad = gradient(x, y, y_prediction)

    # weights updation
    w = w - (learning_rate*grad)

print(f"f(45) after training with {n_iters} epochs: ", forward(45))

w = 0.0
learning_rate = 0.01
n_iters = 50

for epoch in range(n_iters):
    # forward propagation
    y_prediction = forward(x)

    # calculate loss function
    los = loss(y, y_prediction)

    # calculate gradient
    grad = gradient(x, y, y_prediction)

    # weights updation
    w = w - (learning_rate*grad)
    
print(f"f(45) after training with {n_iters} epochs: ", forward(45))