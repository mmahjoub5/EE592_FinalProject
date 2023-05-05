import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ConvBnRelu2d import ConvBnRelu2d


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.interpolate(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x
