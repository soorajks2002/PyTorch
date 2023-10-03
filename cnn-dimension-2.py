import torch

inp = torch.randn(size=(158, 3, 28, 28))

print(inp.shape)


cnn1 = torch.nn.Conv2d(in_channels=3, out_channels=33,
                       kernel_size=(3, 3), stride=1)

out1 = cnn1(inp)

print(out1.shape)


cnn2 = torch.nn.Conv2d(in_channels=33, out_channels=40,
                       kernel_size=(4, 8), stride=1)

out2 = cnn2(out1)

print(out2.shape)


cnn3 = torch.nn.Conv2d(in_channels=40, out_channels=12, kernel_size=(2, 3))

out3 = cnn3(out2)

print(out3.shape)


cnn4 = torch.nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(22, 17))

out4 = cnn4(out3)
print(out4.shape)

#! convert to a 10 dimension feature vector
# out5 = out4.view(10)
# print(out5.shape)


flatten = torch.nn.Flatten()

out5 = flatten(out4)

print(out5.shape)
