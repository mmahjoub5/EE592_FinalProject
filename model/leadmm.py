import torch
import torch.nn as nn
import numpy as np
from .admm import ADMM_Net
import torch.nn.functional as F
from utils.addm_helpers import *
import matplotlib.pyplot as plt
import copy 

class LeADMM(nn.Module):
    def __init__(self, h, iterations) -> None:
        super().__init__()
        self.admm = ADMM_Net(h=h)
        self.admm2 = ADMM_Net(h=h)
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
        w = copy.deepcopy(x)
        self.admm2.restValues()
        self.admm.restValues()
        
        batch_output  = 0

        if x.shape[0] > 1: #if more than one batch
            for b in range(x.shape[0]):
                for i in range(self.iterations):
                    print("iteration: ", i)
                    batch_output = self.leadmm_update(w[b,...], i)
                x[b,:,:] = C(self.admm,batch_output)
                print(x.shape)
                #self.plotBatchImage(x[b,:,:] , "final x leadmm {}".format(b))
                self.admm.restValues()
                self.admm2.restValues()

            return x
        else:
            for i in range(self.iterations):
                print("iteration: ", i)
                self.admm.X = self.leadmm_update(w, i)
                self.plot(self.admm.X, "final x leadmm")
            return C(self.admm, self.admm.X)

        




    def leadmm_update(self, y, i):
        
        self.admm.U  = U_update(self.admm, self.admm.alpha2_k, self.admm.X, self.tau[i].clone(), self.mu2[i].clone())

        # uncropped v update
        self.admm.V = self.admm.V_div_mat * (self.admm.alpha1_k + self.mu1[i].clone() *  H(self.admm.X, self.admm.H_fft) + CT(self.admm, y))
        self.admm.plotImage(self.admm.V, "uncropped v")

        # w update
        old_w = self.admm.W
        zeros = torch.zeros(self.admm.fullSize, dtype = torch.float64, device=self.admm.cuda_device)
        self.admm.W = torch.maximum(self.admm.alpha3_k/self.mu3[i].clone() + self.admm.X, zeros)
        self.admm.plotImage(self.admm.W, "thresholded w")

        # x update (why a conv here????)
        r_k = r_calc(self.admm, self.admm.W, self.admm.V, self.admm.alpha1_k, self.admm.alpha2_k, self.admm.alpha3_k, self.mu1[i].clone(), self.mu2[i].clone(), self.mu3[i].clone(), self.admm.U)
        self.admm.plotImage(r_k, "r_k")
       
        self.admm.X = self.admm.R_div_mat * fft.fft2(fft.ifftshift(r_k))
        self.admm.X = torch.real(fft.fftshift(fft.ifft2(self.admm.X)))
        self.admm.plotImage(self.admm.X, "x update")

        # dual updates/ lagrian
        self.admm.alpha1_k = self.admm.alpha1_k + self.mu1[i].clone() * (H(self.admm.X, self.admm.H_fft) - self.admm.V)
        self.admm.alpha2_k = self.admm.alpha2_k + self.mu2[i].clone() * (Psi(self.admm,self.admm.X) - self.admm.U)
        self.admm.alpha3_k = self.admm.alpha3_k + self.mu3[i].clone() * (self.admm.X - self.admm.W)
        
        return self.admm.X