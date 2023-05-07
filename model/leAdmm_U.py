import torch
import torch.nn as nn
import numpy as np
from .admm import ADMM_Net
import torch.nn.functional as F
from utils.addm_helpers import *
import matplotlib.pyplot as plt
import copy 
from model.leadmm import LeADMM
from model.UNet import UNet270480, UNet_small
from model.admm import ADMM_Net
from utils.addm_helpers import *

class leAdmm_U(nn.Module):
    def __init__(self, h, CNN, iterations, batchSize ) -> None:
        super().__init__() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.admm = ADMM_Net(h=h, batchSize=batchSize, ADMM=True)
        self.admm.alpha2_k = None
        self.iterations = iterations
        self.U_net = CNN
        self.batchSize  = batchSize
        self.admm.U = torch.zeros_like(self.admm.X, dtype = torch.float64, device=self.admm.cuda_device)
    def forward(self, x):
        for i in range(self.iterations):        
            self.admm.X = self.admm_update(x, 0)
            adjust = self.U_net(self.admm.X.unsqueeze(0)).squeeze(0)
            self.admm.U = adjust
        return C(self.admm, self.admm.X)
    
    def admm_update(self, y, batch):
       # import pdb; pdb.set_trace()
        self.admm.V = self.admm.V_div_mat * (self.admm.alpha1_k + self.admm.mu1 *  H(self.admm.X, self.admm.H_fft) + CT(self.admm, y))

        zeros = torch.zeros_like(self.admm.W, dtype = torch.float64, device=self.admm.cuda_device)
        self.admm.W = torch.maximum(self.admm.alpha3_k /self.admm.mu3 + self.admm.X, zeros)
        self.admm.plotImage(self.admm.W, "thresholded w")

         # x update (why a conv here????)
        r_k = LeAmm_r_calc(self.admm, self.admm.W, self.admm.V, self.admm.alpha1_k, self.admm.alpha3_k, self.admm.mu1, self.admm.mu2, self.admm.mu3, self.admm.U)
        self.admm.plotImage(r_k, "r_k")
       
        self.admm.X = self.admm.R_div_mat * fft.fft2(fft.ifftshift(r_k))
        self.admm.X = torch.real(fft.fftshift(fft.ifft2(self.admm.X))).clone()
        self.admm.plotImage(self.admm.X, "x update")

         # dual updates/ lagrian
        # import pdb; pdb.set_trace()
        self.admm.alpha1_k = self.admm.alpha1_k + self.admm.mu1 * (H(self.admm.X, self.admm.H_fft) - self.admm.V)
        self.admm.alpha3_k = self.admm.alpha3_k + self.admm.mu2 * (self.admm.X - self.admm.W)

        return self.admm.X
