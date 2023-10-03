import torch

img1 = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]

t1 = torch.tensor(img1)
print(t1.shape)

img2 = [[[1, 1], [2, 2,], [3, 3]], [
    [4, 4], [5, 5], [6, 6]], [[7, 7], [8, 8], [9, 9]]]

t2 = torch.tensor(img2)
print(t2.shape)

t3 = t2.permute(2, 0, 1)
print(t3)
