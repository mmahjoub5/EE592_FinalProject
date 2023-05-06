import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvBnRelu2d import ConvBnRelu2d

'''
    todo finish this to be able test on cpu!

'''

class CnnDenoiser(nn.Module):
    def __init__(self, inputSize, n_layers) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.inputSize = inputSize
        # want the output size to be the same as the input size
        self.layer0 = ConvBnRelu2d(self.inputSize[0], 32, kernel_size=3, padding=1)
        self.layer1 = ConvBnRelu2d(64, 128, kernel_size=3, padding=1)
        self.layer2 = ConvBnRelu2d(128, 64, kernel_size=3, padding=1)
        self.layer3 = ConvBnRelu2d(64, 32, kernel_size=3, padding=1)
        self.layer4 = ConvBnRelu2d(32, self.inputSize[0], kernel_size=3, padding=1)
    
    def forward(self, x):
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        return x
        
        