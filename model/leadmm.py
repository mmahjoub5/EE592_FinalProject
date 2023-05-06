import torch
import torch.nn as nn
import numpy as np
from .admm import ADMM_Net
import torch.nn.functional as F
from utils.addm_helpers import *
import matplotlib.pyplot as plt
import copy 

class LeADMM(nn.Module):
    def __init__(self, h, iterations, batchSize) -> None:
        super().__init__()
        self.batchSize = batchSize
        self.devioce = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.admm = ADMM_Net(h=h, batchSize=batchSize, ADMM=True)
        self.admm2 = ADMM_Net(h=h, batchSize=batchSize)
        self.iterations = iterations
        #learned parameters
        self.mu1_new = torch.nn.Parameter(self.admm.mu1)
        self.mu2_new = torch.nn.Parameter(self.admm.mu2 )
        self.mu3_new = torch.nn.Parameter(self.admm.mu3)
        self.tau_new = torch.nn.Parameter(self.admm.tau)
    
    def plot(self, image, title):
        plt.figure()
        plt.title(title)
        plt.imshow(image[0,...].detach().numpy(), cmap='gray')
        plt.show()

    def plotBatchImage(self, image, title):
        plt.figure()
        plt.title(title)
        plt.imshow(image.detach().numpy(), cmap='gray')
        plt.show()

    def forward(self, x):
        print(self.mu1_new, self.mu2_new, self.mu3_new, self.tau_new)
        x_k = x.clone()
        for i in range(self.iterations):
            print("iteration: ", i)
            self.admm.X = self.leadmm_update(x_k, i, 0)
       
        return C(self.admm, self.admm.X)
            

    def leadmm_update(self, y, i, batch):
        self.admm.U  = U_update(self.admm, self.admm.alpha2_k, self.admm.X, self.tau_new.clone(), self.mu2_new.clone())

        # uncropped v update
        self.admm.V = self.admm.V_div_mat *  (self.admm.alpha1_k + self.mu1_new.clone() *  H(self.admm.X, self.admm.H_fft) + CT(self.admm, y))
        self.admm.plotImage(self.admm.V, "uncropped v", plot=False)

        # w update
        zeros = torch.zeros_like(self.admm.X, dtype = torch.float64, device=self.admm.cuda_device)
        self.admm.W = torch.maximum(self.admm.alpha3_k/self.tau_new.clone() + self.admm.X, zeros)
        self.admm.plotImage(self.admm.W, "thresholded w leAdmm", plot=False)

        # x update (why a conv here????)
        r_k = r_calc(self.admm, self.admm.W, self.admm.V, self.admm.alpha1_k, self.admm.alpha2_k, self.admm.alpha3_k, self.mu1_new.clone(), self.mu2_new.clone(), self.mu3_new.clone(), self.admm.U)
        self.admm.plotImage(r_k, "r_k leAdmm", plot=False)

        self.admm.X = self.admm.R_div_mat * fft.fft2(fft.ifftshift(r_k))
        self.admm.X = torch.real(fft.fftshift(fft.ifft2(self.admm.X)))
        self.admm.plotImage(self.admm.X, "x update leAdmm", plot=False)

        # dual updates/ lagrian

        self.admm.alpha1_k = self.admm.alpha1_k + self.mu1_new.clone() * (H(self.admm.X, self.admm.H_fft) - self.admm.V)
        self.admm.alpha2_k = self.admm.alpha2_k + self.mu2_new.clone() * (Psi(self.admm,self.admm.X) - self.admm.U)
        self.admm.alpha3_k = self.admm.alpha3_k + self.mu3_new.clone() * (self.admm.X - self.admm.W)
        
        return self.admm.X