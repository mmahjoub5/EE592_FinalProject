import torch
import torch.nn as nn
from utils.addm_helpers import *
import pdb
from torch import fft
import matplotlib.pyplot as plt

class ADMM_Net(nn.Module):
    def __init__(self, h, batchSize= 1, rgb = False, cuda_device = torch.device("cpu"), plotImages = False) -> None:
        super(ADMM_Net, self).__init__()
        self.printstats = True
         ## Initialize constants 
        self.cuda_device = cuda_device
        self.DIMS0 = h.shape[0]  # Image Dimensions
        self.DIMS1 = h.shape[1]  # Image Dimensions
        self.batch_size = batchSize
        self.rgb = rgb
        self.plotImages = plotImages
        if self.rgb:
            self.sensorSize = (self.batch_size, 3, self.DIMS0, self.DIMS1)
            self.fullSize = (self.batch_size, 3, 2 * self.DIMS0, 2 * self.DIMS1)
        else:
            self.sensorSize = (self.batch_size, self.DIMS0, self.DIMS1)
            self.fullSize = (self.batch_size, 2 * self.DIMS0, 2 * self.DIMS1)
            self.stackedShape = (self.batch_size, 2 * self.DIMS0, 2 * self.DIMS1, 2)

        self.h_var = torch.nn.Parameter(torch.tensor(h, dtype=torch.float64, device=cuda_device), requires_grad=False)  # Kernel
        self.h_zeros = torch.nn.Parameter(torch.zeros(self.DIMS0*2, self.DIMS1*2, dtype=torch.float64, device=self.cuda_device),
                                          requires_grad=False)

        self.PAD_SIZE0 = int((self.DIMS0)//2)                           # Pad size to 2* size of PSF
        self.PAD_SIZE1 = int((self.DIMS1)//2)                           # Pad size  to 2 * size of PSF
        ##pdb.set_trace()
        h = torch.tensor(h).view(1, h.shape[0], h.shape[1])
    
        ##### Initialize ADMM variables #####
      
        self.H_fft = torch.fft.fft2(torch.fft.ifftshift(CT(self, h)))

        test = H(torch.ones(self.fullSize), self.H_fft)
        self.X = torch.zeros(self.fullSize, dtype=torch.float64, device=self.cuda_device)
        self.U = torch.zeros((self.fullSize[0], self.fullSize[1], self.fullSize[2], 2), dtype=torch.float64, device=self.cuda_device)
        self.V = torch.zeros(self.fullSize, dtype=torch.float64, device=self.cuda_device)
        self.W = torch.zeros(self.fullSize, dtype=torch.float64, device=self.cuda_device)
        self.alpha1_k = torch.zeros_like(H(self.X, self.H_fft))
        self.alpha2_k = torch.zeros_like(Psi(self, self.X))
        self.alpha3_k = torch.zeros_like(self.W)
       
        # print all the shapes 
        print("H_fft shape: ", self.H_fft.shape)
        print("X shape: ", self.X.shape)
        print("U shape: ", self.U.shape)
        print("V shape: ", self.V.shape)
        print("W shape: ", self.W.shape)
        print("alpha1_k shape: ", self.alpha1_k.shape)
        print("alpha2_k shape: ", self.alpha2_k.shape)
        print("alpha3_k shape: ", self.alpha3_k.shape)
        
        self.LtL = make_laplacian(self)
        self.mu1 = torch.tensor(1e-6, dtype = torch.float64, device=self.cuda_device)
        self.mu2 = torch.tensor(1e-5, dtype = torch.float64, device=self.cuda_device)
        self.mu3 = torch.tensor(1e-5, dtype = torch.float64, device=self.cuda_device)       
        

        self.V_div_mat  = 1./(CT(self, torch.ones(self.sensorSize, dtype=torch.float64, device=self.cuda_device)) + self.mu1)
        self.R_div_mat = self.precompute_R_divmat()
        self.tau = torch.tensor(1e-4, dtype = torch.float64, device=self.cuda_device) * 1000

    
    def plotInput(self, y):
        if self.plotImages:
            plt.figure()
            plt.title("input")
            plt.imshow(y[0,...], cmap='gray')
            plt.show()

    def forward(self, input):
        self.batch_size = input.shape[0]   
        y  = input 
        self.plotInput(y)

        # initial self.X    
        self.iterations = 500
        for i in range(self.iterations):
            self.X  = self.admm_updates(y)
    
        return self.X

    
    def precompute_R_divmat(self):
        HTH = self.mu1 * (np.abs(torch.conj(self.H_fft)* self.H_fft))
        Ltl_compotent = self.mu2 * np.abs(self.LtL)
        return 1./(HTH + Ltl_compotent + self.mu3)

    def plotImage(self, inputImage, title):
        if self.plotImages:
            plt.figure()
            plt.title(title)
            plt.imshow(inputImage[0,...], cmap='gray')
            plt.show()


    # gray scaled/2D image
    def admm_updates(self, y):
        # u update
        self.U  = U_update(self, self.alpha2_k, self.X, self.tau, self.mu2)

        # uncropped v update
        self.V = self.V_div_mat * (self.alpha1_k + self.mu1 *  H(self.X, self.H_fft) + CT(self, y))
        self.plotImage(self.V, "uncropped v")

        # w update
        old_w = self.W
        zeros = torch.zeros(self.fullSize, dtype = torch.float64, device=self.cuda_device)
        self.W = torch.maximum(self.alpha3_k/self.mu3 + self.X, zeros)
        self.plotImage(self.W, "thresholded w")

        # x update (why a conv here????)
        r_k = r_calc(self, self.W, self.V, self.alpha1_k, self.alpha2_k, self.alpha3_k, self.mu1, self.mu2, self.mu3, self.U)
        self.plotImage(r_k, "r_k")
       
        self.X = self.R_div_mat * fft.fft2(fft.ifftshift(r_k))
        self.X = torch.real(fft.fftshift(fft.ifft2(self.X)))
        self.plotImage(self.X, "x update")

        # dual updates/ lagrian
        self.alpha1_k = self.alpha1_k + self.mu1 * (H(self.X, self.H_fft) - self.V)
        self.alpha2_k = self.alpha2_k + self.mu2 * (Psi(self,self.X) - self.U)
        self.alpha3_k = self.alpha3_k + self.mu3 * (self.X - self.W)

        print("U shape: ", self.U.shape)
        print("V shape: ", self.V.shape)
        print("W shape: ", self.W.shape)
        print("X shape: ", self.X.shape)
        print("alpha1_k shape: ", self.alpha1_k.shape)
        print("alpha2_k shape: ", self.alpha2_k.shape)
        print("alpha3_k shape: ", self.alpha3_k.shape)
        #self.r_k = r_calc(self, w_k, v_k, alpha1_k, alpha2_k, alpha3_k, mu1, mu2, mu3, u_k)
        return self.X