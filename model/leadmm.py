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
        self.admm = ADMM_Net(h=h, batchSize=batchSize, cuda_device=self.devioce, ADMM=True)
        self.admm2 = ADMM_Net(h=h, batchSize=batchSize, cuda_device=self.devioce)
        self.iterations = iterations
        #learned parameters
        self.mu1 = torch.nn.Parameter(torch.ones(self.iterations) * self.admm.mu1)
        self.mu2 = torch.nn.Parameter(torch.ones(self.iterations) * self.admm.mu2 )
        self.mu3 = torch.nn.Parameter(torch.ones(self.iterations) * self.admm.mu3)
        self.tau = torch.nn.Parameter(torch.ones(self.iterations) * self.admm.tau)
    
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
        w = x.clone()
        self.admm2.restValues()
        self.admm.restValues()
        # self.admm.printShapes()
        batch_output  = 0

        output = torch.zeros_like(self.admm.X)
        for i in range(self.iterations):
            print("iteration: ", i)
            for batch in range(self.batchSize):
                output[batch,...] = self.leadmm_update(w[batch,...], i, batch)
                
            self.admm.restValues()
                
            # self.admm.printShapes()
        return C(self.admm, output)
            

    def leadmm_update(self, y, i, batch):
        self.admm.U[batch,...]  = U_update(self.admm, self.admm.alpha2_k[batch,...].clone(), self.admm.X[batch,...].clone(), self.tau[i].clone(), self.mu2[i].clone())

        # uncropped v update
        self.admm.V[batch,...] = self.admm.V_div_mat[batch,...].clone() * (self.admm.alpha1_k[batch,...].clone() + self.mu1[i].clone() *  H(self.admm.X[batch,...].clone(), self.admm.H_fft) + CT(self.admm, y))
        self.admm.plotImage(self.admm.V, "uncropped v")

        # w update
        old_w = self.admm.W[batch,...].clone()
        zeros = torch.zeros_like(self.admm.X[batch,...].clone(), dtype = torch.float64, device=self.admm.cuda_device)
        self.admm.W[batch,...] = torch.maximum(self.admm.alpha3_k[batch,...].clone() /self.mu3[i].clone() + self.admm.X[batch,...].clone(), zeros)
        self.admm.plotImage(self.admm.W, "thresholded w")

        # x update (why a conv here????)
        r_k = r_calc(self.admm, self.admm.W[batch,...].clone(), self.admm.V[batch,...].clone(), self.admm.alpha1_k[batch,...].clone(), self.admm.alpha2_k[batch,...].clone(), self.admm.alpha3_k[batch,...].clone(), self.mu1[i].clone(), self.mu2[i].clone(), self.mu3[i].clone(), self.admm.U[batch,...].clone())
        self.admm.plotImage(r_k, "r_k")

        self.admm.X[batch,...] = self.admm.R_div_mat[batch,...].clone() * fft.fft2(fft.ifftshift(r_k))
        self.admm.X[batch,...] = torch.real(fft.fftshift(fft.ifft2(self.admm.X[batch,...].clone())))
        self.admm.plotImage(self.admm.X[batch,...], "x update")

        # dual updates/ lagrian

        self.admm.alpha1_k[batch,...] = self.admm.alpha1_k[batch,...].clone() + self.mu1[i].clone() * (H(self.admm.X[batch,...], self.admm.H_fft) - self.admm.V[batch,...])
        self.admm.alpha2_k[batch,...] = self.admm.alpha2_k[batch,...].clone() + self.mu2[i].clone() * (Psi(self.admm,self.admm.X[batch,...]) - self.admm.U[batch,...])
        self.admm.alpha3_k[batch,...] = self.admm.alpha3_k[batch,...].clone() + self.mu3[i].clone() * (self.admm.X[batch,...] - self.admm.W[batch,...])
        
        return self.admm.X[batch,...]