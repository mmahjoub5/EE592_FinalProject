import numpy as np 
import torch
from utils.addm_helpers import *
import pdb


# gray scaled/2D image
def admm_updates(model, x_k, y, alphak_1, alphak_2, alphak_3, mu1, mu2, mu3, tau):
    # need to update u, v, w, x, alpha1_k, alpha2_k, alpha3_k
    ##pdb.set_trace()
    # u update
    model.U  = U_update(model, alphak_2, x_k, tau, mu2)

   
    # uncropped v update

    v_k = model.V_div_mat * (alphak_1 + mu1 *  H(x_k, model.H_fft) + CT(model, y))

    x_k = model.R_div_mat * r_calc(model, w_k, v_k, alphak_1, alphak_2, alphak_3, mu1, mu2, mu3, u_k)

    # w update
    #pdb.set_trace()
    w_k = torch.maximum(alphak_3/mu3 + x_k, torch.tensor(model.fullSize, dtype = torch.float32, device=model.cuda_device))
    
    # x update
    x_k = model.R_div_mat  * r_calc(model, w_k, v_k, alphak_1, alphak_2, alphak_3, mu1, mu2, mu3, u_k)

    # dual updates/ lagrian
    alphak_1 = alphak_1 + mu1 * (Hadj(model, x_k) - v_k)
    alphak_2 = alphak_2 + mu2 * (Psi(x_k) - u_k)
    alphak_3 = alphak_3 + mu3 * (x_k - w_k)
    r_k = r_calc(model, w_k, v_k, alphak_1, alphak_2, alphak_3, mu1, mu2, mu3, u_k)

def admm(model , in_vars, alpha2k_1, alpha2k_2, CtC, Cty, mu_auto, n, y):  
    ###pdb.set_trace()
    sk = in_vars[0];  alpha1k = in_vars[1]; alpha3k = in_vars[2]
    Hskp = in_vars[3]; 
    
    mu1 = model.mu1[n];  mu2 = model.mu2[n];  mu3 = model.mu3[n]
        
    tau = model.tau[n] #model.mu_vals[3][n]
    
    dual_resid_s = [];  primal_resid_s = []
    dual_resid_u = [];  primal_resid_u = []
    dual_resid_w = []
    primal_resid_w = []
    cost = []

    Smult = 1/(mu1*model.HtH + mu2*model.LtL + mu3)  # May need to expand dimensions 
    Vmult = 1/(CtC + mu1)
    
    ###############  update u = soft(Ψ*x + η/μ2,  tau/μ2) ###################################
    # amin implementation
    u_k  = U_update(0.1, y, mu2, tau)
    print(u_k.shape)
  
    # paper implementation 
    # Lsk1, Lsk2 = L_tf(sk)        # X and Y Image gradients 
    # ukp_1, ukp_2 = soft_2d_gradient2_rgb(model, Lsk1 + alpha2k_1/mu2, Lsk2 + alpha2k_2/mu2, tau)
    
    ################  update      ######################################

    vkp = Vmult*(mu1*(alpha1k/mu1 + Hskp) + Cty)

    ################  update w <-- max(alpha3/mu3 + sk, 0) ######################################


    zero_cuda = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
        
    wkp = torch.max(alpha3k.real/mu3 + sk, zero_cuda, out=None)
   
   
    skp_numerator = mu3*(wkp - alpha3k/mu3) + mu1 * Hadj(model, vkp - alpha1k/mu1) + mu2*Ltv_tf(ukp_1 - alpha2k_1/mu2, ukp_2 - alpha2k_2/mu2) 
    symm = []
    

    SKP_numerator = torch.fft.fft(make_complex(skp_numerator), 2)
    skp = make_real(torch.fft.ifft(complex_multiplication(make_complex(Smult), SKP_numerator), 2))
    
    Hskp_up = Hfor(model, skp)
    r_sv = Hskp_up - vkp
    dual_resid_s.append(mu1 * torch.norm(Hskp - Hskp_up))
    primal_resid_s.append(torch.norm(r_sv))


    if n == model.iterations-1:
        mu1_up = model.mu_vals[0][n]
    else:
        mu1_up = model.mu_vals[0][n+1]

    alpha1kup = alpha1k + mu1*r_sv

    Lskp1, Lskp2 = L_tf(skp)
    r_su_1 = Lskp1 - ukp_1
    r_su_2 = Lskp2 - ukp_2

    dual_resid_u.append(mu2*torch.sqrt(torch.norm(Lsk1 - Lskp1)**2 + torch.norm(Lsk2 - Lskp2)**2))
    primal_resid_u.append(torch.sqrt(torch.norm(r_su_1)**2 + torch.norm(r_su_2)**2))

 
    if n == model.iterations-1:
        mu2_up = model.mu_vals[1][n]
    else:
        mu2_up = model.mu_vals[1][n+1]

    alpha2k_1up= alpha2k_1 + mu2*r_su_1
    alpha2k_2up= alpha2k_2 + mu2*r_su_2

    r_sw = skp - wkp
    dual_resid_w.append(mu3*torch.norm(sk - skp))
    primal_resid_w.append(torch.norm(r_sw))


    if n == model.iterations-1:
        mu3_up = model.mu_vals[2][n]
    else:
        mu3_up = model.mu_vals[2][n+1]

    alpha3kup = alpha3k + mu3*r_sw

    data_loss = torch.norm(C(model, Hskp_up)-y)**2
    tv_loss = tau*TVnorm_tf(skp)

    
    if model.printstats == True:
        
        admmstats = {'dual_res_s': dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_s':  primal_resid_s[-1].cpu().detach().numpy(),
                     'dual_res_w':dual_resid_w[-1].cpu().detach().numpy(),
                     'primal_res_w':primal_resid_w[-1].cpu().detach().numpy(),
                     'dual_res_u':dual_resid_s[-1].cpu().detach().numpy(),
                     'primal_res_u':primal_resid_s[-1].cpu().detach().numpy(),
                     'data_loss':data_loss.cpu().detach().numpy(),
                     'total_loss':(data_loss+tv_loss).cpu().detach().numpy()}
        
        
        print('\r',  'iter:', n,'s:', admmstats['dual_res_s'], admmstats['primal_res_s'], 
         'u:', admmstats['dual_res_u'], admmstats['primal_res_u'],
          'w:', admmstats['dual_res_w'], admmstats['primal_res_w'], end='')
    else:
        admmstats = []

    

    out_vars = torch.stack([skp, alpha1kup, alpha3kup, Hskp_up])

 
    mu_auto_up = torch.stack([mu1_up, mu2_up, mu3_up])
    
    return out_vars, alpha2k_1up, alpha2k_2up, mu_auto_up, symm, admmstats
    
    
