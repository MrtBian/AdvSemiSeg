import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        # shape:
        # input: (N, C_in, H_in, W_in)
        # output: (N, C_out, H_out, W_out)
        # H_{out} = floor((H_{in}+2padding[0]-dilation[0](kernerl_size[0]-1) - 1) / stride[0] + 1)
        # W_{out} = floor((W_{in}+2padding[1]-dilation[1](kernerl_size[1]-1) - 1) / stride[1] + 1)
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
    # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x
