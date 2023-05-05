import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ConvBnRelu2d import ConvBnRelu2d

class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        #import pdb; pdb.set_trace()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small
