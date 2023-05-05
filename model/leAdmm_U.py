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
    def __init__(self, h, iterations, batchSize ) -> None:
        super().__init__() 
        self.admm = ADMM_Net(h=h, batchSize=batchSize)
        self.admm.alpha2_k = None
        self.iterations = iterations
        self.admm.U = torch.zeros_like(self.admm.X)
        self.U_net = UNet_small((1, 540, 960)).double()
        self.batchSize  = batchSize
        self.reshapeForCnn()

    def forward(self, x):
        for i in range(self.iterations):
            self.admm.U = self.U_net(self.admm.X.double())
            for batch in range(self.batchSize):
                self.admm.X[batch,...] = self.admm_update(x[batch,...], batch= batch)
        self.admm.restValues()
        self.reshapeForCnn()
        return C(self.admm, self.admm.X)
        
    def reshapeForCnn(self):
        self.admm.X = self.admm.X.view(self.batchSize, 1, 540, 960)
        self.admm.U = self.admm.U.view(self.batchSize, 1, 540, 960)
        self.admm.V = self.admm.V.view(self.batchSize, 1, 540, 960)
        self.admm.W = self.admm.W.view(self.batchSize, 1, 540, 960)
        self.admm.alpha1_k = self.admm.alpha1_k.view(self.batchSize, 1, 540, 960)
        self.admm.alpha3_k = self.admm.alpha3_k.view(self.batchSize, 1, 540, 960)
    
    def admm_update(self, y, batch):
  
        self.admm.V[batch,...] = self.admm.V_div_mat[batch,...] * (self.admm.alpha1_k[batch,...] + self.admm.mu1 *  H(self.admm.X[batch,...], self.admm.H_fft) + CT(self.admm, y))

        zeros = torch.zeros_like(self.admm.W[batch,...], dtype = torch.float64, device=self.admm.cuda_device)
        self.admm.W[batch,...] = torch.maximum(self.admm.alpha3_k[batch,...] /self.admm.mu3 + self.admm.X[batch,...], zeros)
        self.admm.plotImage(self.admm.W, "thresholded w")

         # x update (why a conv here????)
        r_k = LeAmm_r_calc(self.admm, self.admm.W[batch,...], self.admm.V[batch,...], self.admm.alpha1_k[batch,...], self.admm.alpha3_k[batch,...], self.admm.mu1, self.admm.mu2, self.admm.mu3, self.admm.U[batch,...])
        self.admm.plotImage(r_k, "r_k")
       
        self.admm.X[batch,...] = self.admm.R_div_mat[batch,...] * fft.fft2(fft.ifftshift(r_k))
        self.admm.X[batch,...] = torch.real(fft.fftshift(fft.ifft2(self.admm.X[batch,...])))
        self.admm.plotImage(self.admm.X, "x update")

         # dual updates/ lagrian
        # import pdb; pdb.set_trace()
        self.admm.alpha1_k[batch,...] = self.admm.alpha1_k[batch,...] + self.admm.mu1 * (H(self.admm.X[batch,...], self.admm.H_fft) - self.admm.V[batch,...])
        self.admm.alpha3_k[batch,...] = self.admm.alpha3_k[batch,...] + self.admm.mu2 * (self.admm.X[batch,...] - self.admm.W[batch,...])

        return self.admm.X[batch,...]
