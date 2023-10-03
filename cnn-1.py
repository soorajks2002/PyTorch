import torch


class PracticeNET(torch.nn.Module):
    def __init__(self, img_channel=3, embed_size=10):
        super(PracticeNET, self).__init__()

        self.con1 = torch.nn.Conv2d(
            in_channels=img_channel, out_channels=33, kernel_size=(3, 3))
        self.con2 = torch.nn.Conv2d(
            in_channels=33, out_channels=40, kernel_size=(4, 8), stride=1)
        self.con3 = torch.nn.Conv2d(
            in_channels=40, out_channels=12, kernel_size=(2, 3))
        self.con4 = torch.nn.Conv2d(
            in_channels=12, out_channels=embed_size, kernel_size=(22, 17))

        self.flatten = torch.nn.Flatten()

    def __call__(self, x):
        out = self.con1(x)
        out = self.con2(out)
        out = self.con3(out)
        out = self.con4(out)
        out = self.flatten(out)

        return out


# embedd image to a 10 sized vector

n_images = 130
n_channels = 3  # 3 for 'R' 'G' 'B'
img_w = 28
img_h = 28

image_embedding_size = 10

x = torch.randn(size=(n_images, n_channels, img_h, img_w))

model = PracticeNET(n_channels, image_embedding_size)

output = model(x)

print(output.shape)
