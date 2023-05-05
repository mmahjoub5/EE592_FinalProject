import numpy as np 
import torch
import torch.nn.functional as F
from torch import fft

## 2D case 
def H(xk, H_fft): 
    #print(xk)
    # import pdb; pdb.set_trace()
    return torch.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(xk))*H_fft)))
def HT(x, H_fft):
    x_zeroed = fft.ifftshift(x)
    return torch.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * torch.conj(H_fft))))


def CT(model, x):
    #print(model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)

def C(model, x):
    if model.rgb:
        C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # C indices 
        C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # C indices 
        return x[:, :, C01:C02, C11:C12]
    else:
        import pdb; ##pdb.set_trace()
        top = (model.fullSize[-2] - model.sensorSize[-2])//2
        bottom = (model.fullSize[-2] + model.sensorSize[-2])//2
        left = (model.fullSize[-1] - model.sensorSize[-1])//2
        right = (model.fullSize[-1] + model.sensorSize[-1])//2
        return x[...,top:bottom,left:right]

def pad_zeros_torch(model, x):
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)


###### Things I saw on TV ###########
def make_laplacian(model):
    lapl = torch.zeros([model.batch_size, model.DIMS0*2,model.DIMS1*2])
    lapl[:,0,0] =4.; 
    lapl[:,0,1] = -1.; lapl[:,1,0] = -1.; 
    lapl[:,0,-1] = -1.; lapl[:,-1,0] = -1.; 

    LTL = torch.fft.fft2(lapl)
    return LTL

#######Soft Thresholding########
def Psi(model, v):
    if model.rgb: #most likely wrong for rgb
        temp = torch.roll(v,1,dims=0) - v 
        temp2 = torch.roll(v,1,dims=1) - v
        temp3 = torch.stack((temp, temp2), dim=4)
    else:
        temp = torch.roll(v,1,dims=-2) - v 
        temp2 = torch.roll(v,1,dims=-1) - v
        temp3 = torch.stack((temp, temp2), dim=-1)
        # assert temp3.shape == model.stackedShape
    return temp3

def SoftThresholding(x, tau):
    z = torch.zeros_like(x, dtype=torch.float32)
    return torch.sign(x) * torch.maximum(z, torch.abs(x) - tau)

######Psi ########
def PsiT(model, u):
    if model.rgb:
        diff1 = torch.roll(u[..., ..., ..., ..., 0], -1, dims=0) - u 
        diff2 = torch.roll(u[..., ..., ..., ..., 1], -1, dims=1) - u
    else:
        diff1 = torch.roll(u[..., 0], -1, dims=-2) - u[...,0]
        diff2 = torch.roll(u[..., 1], -1, dims=-1) - u[...,1]
    return diff1 + diff2

####### ADMM Updates #######
def U_update(model, eta, image_est, tau, mu2):
    temp = Psi(model, image_est) + eta/mu2
    return SoftThresholding(Psi(model, image_est) + eta/mu2, tau/mu2)



# residual update 
def r_calc(model, w_k, v_k, alpha1_k, alpha2_k, alpha3_k , mu_1, mu_2, mu_3, u_k):
    return (mu_3 * w_k - alpha3_k) + PsiT(model, mu_2* u_k  - alpha2_k) + (HT(mu_1 * v_k - alpha1_k, model.H_fft))

def LeAmm_r_calc(model, w_k, v_k, alpha1_k, alpha3_k , mu_1, mu_2, mu_3, u_k):
    return (mu_3 * w_k - alpha3_k) + mu_2* u_k + (HT(mu_1 * v_k - alpha1_k, model.H_fft))

def X_update(w, alphak_3, u, alphak_2, x, alphak_1, H_fft, R_divmat ):
    freq_space_result = R_divmat* torch.fft.fft2( torch.fft.ifftshift(r_calc(w, alphak_3, u, alphak_2, x, alphak_1, H_fft)) )
    return torch.real(torch.fft.fftshift(torch.fft.ifft2(freq_space_result)))



######## normalize image #########
def normalize_image(image):
    out_shape = image.shape
    image_flat = image.reshape((out_shape[0],out_shape[1]*out_shape[2]*out_shape[3]))
    image_flat = image_flat.float()
    image_max,_ = torch.max(image_flat,1)
    image_max_eye = torch.eye(out_shape[0], dtype = torch.float32, device=image.device)*1/image_max
    image_normalized = torch.reshape(torch.matmul(image_max_eye, image_flat), (out_shape[0],out_shape[1],out_shape[2],out_shape[3]))
    
    return image_normalized
