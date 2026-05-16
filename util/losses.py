from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

def rrc_loss_infonce(
    sim,       
    attn,      
    density,    
    attn_thr=1,
    den_thr=1e-3 * 60,
    tau=0.07,   
    eps=1e-6
):
    

    B, N = sim.shape
    h = w = int(sqrt(N))
    sim = sim.to(density.dtype)
    device, dtype = sim.device, sim.dtype


    attn_det = attn.detach()
    attn_max, _ = attn_det.max(dim=1, keepdim=True)
    attn_min, _ = attn_det.min(dim=1, keepdim=True)
    attn_norm = (attn_det - attn_min) / (attn_max - attn_min + eps)   

    attn_2d = attn_norm.view(B, 1, h, w)                              
    AN = (attn_2d >= attn_thr)                                      
    # AN = ~P


    kernel = torch.ones(1, 1, 16, 16, device=device, dtype=dtype)
    gt_patch = F.conv2d(density.unsqueeze(1), kernel, stride=16)    
    P = (gt_patch >= den_thr)                                     
    
  
    pos_mask = P.view(B, -1)                                                                  
    neg_mask = (~P).view(B, -1)
    
    all_mask = pos_mask | neg_mask                                  

  
    logits = sim                                           
    
    very_neg = torch.full_like(logits, -1e9)                     

    logits_pos = torch.where(pos_mask, logits, very_neg)             
    log_num = torch.logsumexp(logits_pos, dim=1)                     


    logits_all = torch.where(all_mask, logits, very_neg)
    log_den = torch.logsumexp(logits_all, dim=1)                    

    loss_vec = -(log_num - log_den)                                

    pos_count = pos_mask.sum(dim=1)
    neg_count = neg_mask.sum(dim=1)
    valid = (pos_count > 0) & (neg_count > 0)

    if valid.any():
        loss = loss_vec[valid].mean()
    else:
        loss = torch.tensor(0.0, device=device, dtype=dtype)

    return loss
