import numpy as np 
import torch
import torch.nn.functional as F
from torch import fft

## 2D case 
## TODO: 
def H(xk, H_fft): 
    print(xk)
    return torch.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(xk))*H_fft)))
def HT(x, H_fft):
    x_zeroed = fft.ifftshift(x)
    return torch.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * torch.conj(H_fft))))


def CT(model, x):
    print(model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)

def C(model, x):
    if model.rgb:
        C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # C indices 
        C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # C indices 
        return x[:, :, C01:C02, C11:C12]
    else:
        import pdb; ##pdb.set_trace()
        top = (model.fullSize[1] - model.sensorSize[1])//2
        bottom = (model.fullSize[1] + model.sensorSize[1])//2
        left = (model.fullSize[2] - model.sensorSize[2])//2
        right = (model.fullSize[2] + model.sensorSize[2])//2
        return x[:, top:bottom,left:right]

def pad_zeros_torch(model, x):
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return F.pad(x, PADDING, 'constant', 0)

####### FFT Shifting #####
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):  #loop backwards
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


###### Complex operations ##########
def complex_multiplication(t1, t2):
    real1, imag1 = torch.unbind(t1, dim=-1)
    real2, imag2 = torch.unbind(t2, dim=-1)
    
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

def complex_abs(t1):
    real1, imag1 = torch.unbind(t1, dim=2)
    return torch.sqrt(real1**2 + imag1**2)

def make_real(c):
    out_r, _ = torch.unbind(c,-1)
    return out_r

def make_complex(r, i = 0):
    if i==0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.stack((r, i), -1)


###### Things I saw on TV ###########
def make_laplacian(model):
    lapl = torch.zeros([model.batch_size, model.DIMS0*2,model.DIMS1*2])
    lapl[:,0,0] =4.; 
    lapl[:,0,1] = -1.; lapl[:,1,0] = -1.; 
    lapl[:,0,-1] = -1.; lapl[:,-1,0] = -1.; 

    LTL = torch.fft.fft2(lapl)
    return LTL

####### Forward Model #####

def Hfor(model, x):
    xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
    #X = torch.fft(batch_ifftshift2d(xc),2)
    X = torch.fft.fft(xc,2)
    HX = complex_multiplication(model.H,X)
    out = torch.fft.ifft(HX,2)
    out_r, _ = torch.unbind(out,-1)
    return out_r

def Hadj(model, x):
    xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
    #X = torch.fft(batch_ifftshift2d(xc),2)
    X = torch.fft.fft(xc,2)
    HX = complex_multiplication(model.Hconj,X)
    #out = batch_ifftshift2d(torch.ifft(HX,2))
    out = torch.fft.ifft(HX,2)
    out_r, _ = torch.unbind(out,-1)
    return out_r

#######Soft Thresholding########
def Psi(model, v):
    # import pdb; ##pdb.set_trace()
   
    if model.rgb: #most likely wrong for rgb
        temp = torch.roll(v,1,dims=0) - v 
        temp2 = torch.roll(v,1,dims=1) - v
        temp3 = torch.stack((temp, temp2), dim=4)
    else:
        temp = torch.roll(v,1,dims=1) - v 
        temp2 = torch.roll(v,1,dims=2) - v
        temp3 = torch.stack((temp, temp2), dim=3)
        assert temp3.shape == model.stackedShape
    return temp3

def SoftThresholding(x, tau):
    z = torch.zeros_like(x, dtype=torch.float32)
    return torch.sign(x) * torch.maximum(z, torch.abs(x) - tau)

######Psi ########
def PsiT(model, u):
    import pdb; #pdb.set_trace()
    if model.rgb:
        diff1 = torch.roll(u[..., ..., ..., ..., 0], -1, dims=0) - u 
        diff2 = torch.roll(u[..., ..., ..., ..., 1], -1, dims=1) - u
    else:
        diff1 = torch.roll(u[:, :, :, 0], -1, dims=1) - u[...,0]
        diff2 = torch.roll(u[:, :, :, 1], -1, dims=2) - u[...,1]
    return diff1 + diff2

####### ADMM Updates #######
def U_update(model, eta, image_est, tau, mu2):
    # import pdb; #pdb.set_trace()
    temp = Psi(model, image_est) + eta/mu2

    return SoftThresholding(Psi(model, image_est) + eta/mu2, tau/mu2)



# residual update 
def r_calc(model, w_k, v_k, alpha1_k, alpha2_k, alpha3_k , mu_1, mu_2, mu_3, u_k):
    return (mu_3 * w_k - alpha3_k) + PsiT(model, mu_2* u_k  - alpha2_k) + (HT(mu_1 * v_k - alpha1_k, model.H_fft))

def X_update(w, alphak_3, u, alphak_2, x, alphak_1, H_fft, R_divmat ):
    freq_space_result = R_divmat* torch.fft.fft2( torch.fft.ifftshift(r_calc(w, alphak_3, u, alphak_2, x, alphak_1, H_fft)) )
    return torch.real(torch.fft.fftshift(torch.fft.ifft2(freq_space_result)))
#######       #######


def TVnorm_tf(x):
    x_diff, y_diff = L_tf(x)
    result = torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
    return result

def L_tf(a): # Not using
    xdiff = a[:,:, 1:, :]-a[:,:, :-1, :]
    ydiff = a[:,:, :, 1:]-a[:,:, :, :-1]
    return -xdiff, -ydiff


def Ltv_tf(a, b): # Not using
    return torch.cat([a[:,:, 0:1,:], a[:,:, 1:, :]-a[:,:, :-1, :], -a[:,:,-1:,:]],
                2) + torch.cat([b[:,:,:,0:1], b[:, :, :, 1:]-b[:, :, :,  :-1], -b[:,:, :,-1:]],3)
    #return tf.concat([a[:,0:1,:], a[:, 1:, :]-a[:, :-1, :], -a[:,-1:,:]], axis = 1) + tf.concat([b[:,:,0:1], b[:, :, 1:]-b[:, :,  :-1], -b[:,:,-1:]], axis = 2)


######## ADMM Parameter Update #########
def param_update_previous(mu, res_tol, mu_inc, mu_dec, r, s):
    
    if r > res_tol * s:
        mu_up = mu*mu_inc
    if s > res_tol*s:
        mu_up = mu/mu_dec
    else:
        mu_up = mu
   
    #mu_up = tf.cond(tf.greater(r, res_tol * s), lambda: (mu * mu_inc), lambda: mu)
    #mu_up = tf.cond(tf.greater(s, res_tol * r), lambda: (mu_up/mu_dec), lambda: mu_up)
    
    return mu_up

######## ADMM Parameter Update #########
def param_update2(mu, res_tol, mu_inc, mu_dec, r, s):
    
    if r > res_tol * s:
        mu_up = mu*mu_inc
    else:
        mu_up = mu
        
    if s > res_tol*r:
        mu_up = mu_up/mu_dec
    else:
        mu_up = mu_up
   
    #mu_up = tf.cond(tf.greater(r, res_tol * s), lambda: (mu * mu_inc), lambda: mu)
    #mu_up = tf.cond(tf.greater(s, res_tol * r), lambda: (mu_up/mu_dec), lambda: mu_up)
    
    return mu_up


####### Soft Thresholding Functions  #####

def soft_2d_gradient2_rgb(model, v,h,tau):
    z0 = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
    z1 = torch.zeros(model.batch_size, 3, 1, model.DIMS1*2, dtype = torch.float32, device=model.cuda_device)
    z2 = torch.zeros(model.batch_size, 3, model.DIMS0*2, 1, dtype= torch.float32, device=model.cuda_device)

    vv = torch.cat([v, z1] , 2)
    hh = torch.cat([h, z2] , 3)
    import pdb; ###pdb.set_trace()
    mag = torch.sqrt(vv*vv + hh*hh)

    print(mag.type())

    magt = torch.max(mag - tau, z0, out=None)
    mag = torch.max(mag - tau, z0, out=None) + tau
    #smax = torch.nn.Softmax()
    #magt = smax(mag - tau, torch.zeros_like(mag, dtype = torch.float32))
    #mag = smax(mag - tau, torch.zeros_like(mag, dtype = torch.float32)) + tau
    mmult = magt/(mag)#+1e-5)
    if torch.any(mmult != mmult):
        print('here')
    if torch.any(v != v):
        print('there')

    return v*mmult[:,:, :-1,:], h*mmult[:,:, :,:-1]

######## normalize image #########
def normalize_image(image):
    out_shape = image.shape
    image_flat = image.reshape((out_shape[0],out_shape[1]*out_shape[2]*out_shape[3]))
    image_flat = image_flat.float()
    image_max,_ = torch.max(image_flat,1)
    image_max_eye = torch.eye(out_shape[0], dtype = torch.float32, device=image.device)*1/image_max
    image_normalized = torch.reshape(torch.matmul(image_max_eye, image_flat), (out_shape[0],out_shape[1],out_shape[2],out_shape[3]))
    
    return image_normalized
