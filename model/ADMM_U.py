from model.admm import ADMM_Net
from model.UNet import UNet_small
import torch
import torch.nn as nn

class ADMM_U(nn.Module):
    def __init__(self, h , ) -> None:
        super().__init__()
        self.admm = ADMM_Net(h=h, batchSize=1, ADMM=True, iterations=50)
        self.U_net = UNet_small((1, 540, 960)).double()

    def forward(self, x):
        x = self.admm(x)
        x = self.U_net(x.view(1, 1, 270, 480).double())
        return x