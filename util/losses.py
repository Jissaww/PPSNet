from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


# 前两个是通过attn weight去约束预测密度图
def rank_loss(attn, density, K=64, margin=0):
    # attn: [B, num_heads, Nx+Ny, Nx+Ny]
    # density: [B, 384, 384]
    
    b, h, w = density.shape
    Nx = 576

    attn = attn.mean(dim=1)

    # attn:[B, 576]
    attn = attn[:, Nx:, :Nx].mean(dim=1).detach()
    
    # 为了保证确定性
    density = density.view(b, 24, 16, 24, 16)
    density = density.mean(dim=(2, 4))  
    # density = F.adaptive_avg_pool2d(density.unsqueeze(1), output_size=(24, 24))  

    density = density.view(b, -1)

    losses = []

    for i in range(b):
        
        attn_i = attn[i]
        density_i = density[i]

        top_vals, top_idx = attn_i.topk(K, largest=True)
        low_vals, low_idx = attn_i.topk(K, largest=False)

        d_top = density_i[top_idx]
        d_low = density_i[low_idx]

        loss_ij = F.relu(margin - (d_top - d_low))
        losses.append(loss_ij.mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=density.device)
    
    loss_rank = torch.stack(losses).mean()

    return loss_rank


# 试了权重为1--train上指标不变, 0.001
def rank_lossV2(attn, density, K=64, margin=0, l=8):
    # attn: [B, num_heads, Nx+Ny, Nx+Ny]
    # density: [B, 384, 384]
    
    b, h, w = density.shape
    Nx = 576

    attn = attn.mean(dim=1)

    # attn:[B, 576]
    attn = attn[:, Nx:, :Nx].mean(dim=1).detach()
    
    # 为了保证确定性
    density = density.view(b, 24, 16, 24, 16)
    density = density.mean(dim=(2, 4))  
    # density = F.adaptive_avg_pool2d(density.unsqueeze(1), output_size=(24, 24))  

    density = density.view(b, -1)
    losses = []

    for i in range(b):
        attn_i = attn[i]
        density_i = density[i]

        _, sorted_idx = torch.sort(attn_i, descending=True)

        # for j in range(0, Nx-l):
        #     idx_high = sorted_idx[j]
        #     idx_low = sorted_idx[j+l]

        #     density_high = density_i[idx_high]
        #     density_low = density_i[idx_low]

        #     loss_ij = F.relu(margin - (density_high - density_low))
        #     losses.append(loss_ij)

        high_idx = sorted_idx[:-l]
        low_idx = sorted_idx[l:]

        density_high = density_i[high_idx]
        density_low = density_i[low_idx]

        loss_ij = F.relu(margin - (density_high - density_low))
        losses.append(loss_ij.mean())


    if len(losses) == 0:
        return torch.tensor(0, device=attn.device)
    
    rank_loss = torch.stack(losses).mean()

    return rank_loss


# V3是通过GT去约束密度图, 利用滑动窗口的思想进行约束
def rank_lossV3(density, gt_density, patch_size=16, stride=16, margin=0, step=1):
    # step: GT和pred比较之间的间隔

    def slide_patches(density, patch_size=16, stride=16):
        # density: [B, 384, 384]
        # density: [B, 1, 384, 384]
        density = density.unsqueeze(1)
        b, c, h, w = density.shape

        # patches: [B, 256, 576] [B, C*patch_size*patch_size, num_patches]
        patches = F.unfold(density, kernel_size=patch_size, stride=stride)

        # patches_count: [B, 576]
        patches_count = patches.sum(dim=1)/60

        return patches_count
    

    # pred/gt_count: [B, 576]
    pred_count = slide_patches(density, patch_size=patch_size, stride=stride)
    gt_count = slide_patches(gt_density, patch_size=patch_size, stride=stride)
    
    b, num_patches = pred_count.shape

    losses = []

    for i in range(b):
        gt_count_i = gt_count[i]
        pred_count_i = pred_count[i]

        _, sorted_idx = torch.sort(gt_count_i, descending=True)
        high_idx = sorted_idx[:-step]
        low_idx = sorted_idx[step:]
        
        pred_high = pred_count_i[high_idx]
        pred_low = pred_count_i[low_idx]

        loss_ij = F.relu(margin - (pred_high - pred_low))
        losses.append(loss_ij.mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_count.device)
    
    return torch.stack(losses).mean()


def count_loss(density, gt_density, K=8):
    # K表示划分成 K*K 个格子
    B, H, W = density.shape
    h = int(H // K)
    w = int(W // K)

    local_avg = F.avg_pool2d(
        density, kernel_size=(h, w), stride=(h, w)
    )
    local_sum = local_avg
    
    gt_avg = F.avg_pool2d(
        gt_density, kernel_size=(h, w), stride=(h, w)
    )
    gt_sum = gt_avg

    local_count_loss = ((local_sum - gt_sum) ** 2).mean()

    return local_count_loss


# 结合BMNet和t2i的损失
def attnrrc_loss(sim, pt_map):
    # sim: [b, 576]
    # pt_map: [b, 1, 384, 384]
    kernel = torch.ones(1, 1, 16, 16).to(sim.device)
    # pt_map [b, 1, 24, 24]
    pt_map = F.conv2d(pt_map.float(), kernel, stride=16).bool()

    b, _, h, w = pt_map.shape
    sim = sim.view(b, 1, h, w)
    sim = torch.sigmoid(sim)


    pos_map = (1 - sim) * pt_map
    neg_map = torch.clamp(sim, min=0) * (~pt_map)

    pos = pos_map.flatten(1).sum(dim=1)
    neg = neg_map.flatten(1).sum(dim=1)

    pos_num = pt_map.flatten(1).sum(dim=1)
    neg_num = (~pt_map).flatten(1).sum(dim=1)
    
    loss = pos / (pos_num + 1e-6) + neg / (neg_num + 1e-6)

    print("pos:", (pos / (pos_num + 1e-6)).mean().item(), "neg:", (neg / (neg_num + 1e-6)).mean().item())

    return loss.mean()


# BMNet中的loss
def attn_loss(attn, pt_map):
    # attn:[b, 1, 24, 24]
    # pt_map: [b, 1, 384, 384]

    kernel = torch.ones(1, 1, 16, 16).to(attn.device)
    # pt_map: [b, 1, 24, 24]
    pt_map = F.conv2d(pt_map.float(), kernel, stride=16).bool()
   
    b, _, h, w = pt_map.shape
    # pt_map: [b, 576]
    pt_map = pt_map.view(b, -1)
    
    attn = attn.view(b, -1)

    scale = 2.0
    attn = torch.exp(attn * scale)

    loss = 0
    bs = 0
    for idx in range(b):
        if pt_map[idx].sum() == 0:
            continue
        pos_attn = attn[idx][pt_map[idx]].sum()
        neg_attn = attn[idx][~pt_map[idx]].sum()

        sample_loss = -1 * torch.log(pos_attn / (neg_attn + pos_attn + 1e-6))

        loss += sample_loss
        bs = bs + 1
    
    if bs == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    else:
        return loss / bs


# loss decay
def attn_loss_decay(attn, pt_map):
    # attn:[b, 1, 24, 24]
    # pt_map: [b, 1, 384, 384]

    kernel = torch.ones(1, 1, 16, 16).to(attn.device)
    # pt_map: [b, 1, 24, 24]
    pt_map = F.conv2d(pt_map.float(), kernel, stride=16).bool()
   
    b, _, h, w = pt_map.shape
    # pt_map: [b, 576]
    pt_map = pt_map.view(b, -1)
    
    attn = attn.view(b, -1)

    scale = 1.0
    attn = torch.exp(attn * scale)

    bg_decay = 0.2

    loss = 0
    bs = 0
    for idx in range(b):
        if pt_map[idx].sum() == 0:
            continue
        pos_attn = attn[idx][pt_map[idx]].sum()
        neg_attn = attn[idx][~pt_map[idx]].sum()

        sample_loss = -1 * torch.log(pos_attn / (neg_attn * bg_decay + pos_attn + 1e-6))

        loss += sample_loss
        bs = bs + 1
    
    if bs == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    else:
        return loss / bs


# 根据密度加权
def attn_loss_weight(attn, pt_map):
    # attn: [b, 1, 24, 24]
    # pt_map: [b, 1, 384  384]

    b, _, h, w = pt_map.shape
    kernel = torch.ones(1, 1, 16, 16).to(pt_map.device)
    points = F.conv2d(pt_map.float(), kernel, stride=16)
    pt_map = F.conv2d(pt_map.float(), kernel, stride=16).bool()
    
    weights = points * 0.5  + 1
    # max_points = 16 * 16
    # density = pt_map / max_points
    # weights = 1.0 + density * 2.0

    # print(density[0][0][0])
    # print(pt_map[0][0][0])

    attn_flat = attn.view(b, -1)
    pt_flat = pt_map.view(b, -1)
    weights_flat = weights.view(b, -1)
    
    scale = 1.0
    attn_flat = torch.exp(attn_flat*scale)

    loss = 0
    bs = 0
    for idx in range(b):
        if pt_flat[idx].sum() == 0:
            continue

        pos_attn = attn_flat[idx][pt_flat[idx]]
        pos_weights = weights_flat[idx][pt_flat[idx]]

        weights_pos_attn = (pos_attn * pos_weights).sum()
        neg_attn = attn_flat[idx][~pt_flat[idx]].sum()

        sample_loss = -1 * torch.log(weights_pos_attn / (neg_attn + weights_pos_attn + 1e-6))

        loss += sample_loss
        bs = bs + 1

    if bs == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    else:
        return loss / bs


# 分区选择pt_map
def attn_lossV2(attn, pt_map):
    # attn:[b, 1, 24, 24]
    # pt_map: [b, 1, 384, 384]

    kernel = torch.ones(1, 1, 16, 16).to(attn.device)
    # pt_map: [b, 1, 24, 24]
    pt_map = F.conv2d(pt_map.float(), kernel, stride=16)

        
    b, _, h, w = pt_map.shape
    # pt_map: [b, 576]
    pt_map = pt_map.view(b, -1)
    
    attn = attn.view(b, -1)
    attn = torch.exp(attn)


    loss = 0
    bs = 0
    for idx in range(b):

        cnt = pt_map[idx]

        if cnt.sum() == 0:
            continue

        bg_mask = (cnt == 0)
        strong_mask = (cnt >= 2)
        weak_mask = (cnt == 1)

        # if strong_mask.sum() == 0:
        #     fg_mask= (cnt >= 1)

        #     pos_attn = attn[idx][fg_mask].sum()
        #     neg_attn = attn[idx][bg_mask].sum()
        #     sample_loss = -1 * torch.log(pos_attn / (pos_attn + neg_attn + 1e-6))
        #     loss += sample_loss
        #     bs = bs + 1
        #     continue
            
        pos_attn = 2 * attn[idx][strong_mask].sum() + attn[idx][weak_mask].sum()
        neg_attn = attn[idx][bg_mask].sum()
        sample_loss = -1 * torch.log(pos_attn / (neg_attn + pos_attn + 1e-6))
        loss += sample_loss
        bs = bs + 1

    if bs == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    else:
        return loss / bs


# 选k个(效果不好)
def attn_lossV3(attn, pt_map, neg_ratio=0.10):
    # attn:[b, 1, 24, 24]
    # pt_map: [b, 1, 384, 384]

    kernel = torch.ones(1, 1, 16, 16).to(attn.device)
    # pt_map: [b, 1, 24, 24]
    pt_map = F.conv2d(pt_map.float(), kernel, stride=16).bool()
        
    b, _, h, w = pt_map.shape
    # pt_map: [b, 576]
    pt_map = pt_map.view(b, -1)
    attn = attn.view(b, -1)

    attn = torch.exp(attn)

    loss = 0
    bs = 0

    for idx in range(b):
        fg_mask = pt_map[idx]
        if fg_mask.sum() == 0:
            continue

        bg_mask = ~fg_mask

        fg_scores = attn[idx][fg_mask]
        bg_scores = attn[idx][bg_mask]
        bgl = bg_scores.numel()
        if bgl == 0:
            continue

        k_neg = min(bgl, max(1, int(bg_scores.numel() * neg_ratio)))
        # print(k_neg, bgl)
        hard_neg_scores, _ = torch.topk(bg_scores, k_neg, largest=True)
        # print(hard_neg_scores.shape, bg_scores.shape)
        
        neg_hard = hard_neg_scores.sum()
        pos_hard = fg_scores.sum()

        sample_loss = -1 * torch.log(pos_hard / (neg_hard + pos_hard + 1e-6))
        loss += sample_loss
        bs = bs + 1

    if bs == 0:
        return torch.tensor(0.0, device=attn.device, dtype=attn.dtype)
    else:
        return loss / bs
        

# attn_loss的变体，使用了logsumexp避免爆0
def sim_loss(sim, pt_map):
    # sim:   [B, 1, 24, 24]  相似度 logits（可以正负、值可大可小）
    # pt_map:[B, 1, 384,384]

    B = sim.shape[0]
    kernel = torch.ones(1, 1, 16, 16, device=sim.device, dtype=sim.dtype)
    mask = F.conv2d(pt_map.float(), kernel, stride=16) > 0   # [B,1,24,24] bool

    sim  = sim.view(B, -1)   # [B,576]
    mask = mask.view(B, -1)  # [B,576]

    loss = sim.new_tensor(0.0)
    bs   = 0

    for i in range(B):
        m = mask[i]
        if m.sum() == 0 or (~m).sum() == 0:
            continue

        pos_logits = sim[i][m]
        neg_logits = sim[i][~m]

        log_pos = torch.logsumexp(pos_logits, dim=0)                           # log Σ e^{Sp}
        log_all = torch.logsumexp(torch.cat([pos_logits, neg_logits]), dim=0)  # log Σ e^{S}

        sample_loss = -(log_pos - log_all)
        loss += sample_loss
        bs   += 1

    if bs == 0:
        return sim.sum() * 0.0
    return loss / bs


# vlcounter中的loss
def rank_loss(density, sim):
    # density: [B, 384, 384]
    # sim: [B, 1, 576]
    b, n = sim.shape
    h = w = int(sqrt(n))
    sim = sim.view(b, 1, h, w)

    # b, _, h, w = sim.shape
    mask = F.interpolate(density.unsqueeze(1), size=(h, w), mode='bilinear')
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    # print(mask.max(), mask.min(), mask.mean()) 

    infonce = 0
    rank = [1.0, 0.8, 0.6, 0.4]
    
    for i in range(3):
        r_mask = torch.where((mask > rank[i+1]) & (mask < rank[i]), 1.0, 0.0)
        inv_mask = torch.where(mask < rank[i+1], 1.0, 0.0)

        pos = torch.sum(torch.exp(sim * r_mask), dim=(1, 2, 3))
        neg = torch.sum(torch.exp(sim * inv_mask), dim=(1, 2, 3))

        infonce += torch.mean(-torch.log(pos / (pos + neg + 1e-6)))

    return infonce


# 原始的rcc损失
def rrc_loss(sim, attn, density, attn_thr=0.3, den_thr=1e-3*60):
    # sim: [B, 576]
    # attn: [B, 576]
    # density: [B, 384, 384]
    b, n = sim.shape
    h = w = int(sqrt(n))

    sim = torch.sigmoid(sim)
    attn_max = attn.max(dim=1, keepdim=True)[0]
    attn_min = attn.min(dim=1, keepdim=True)[0]
    attn = (attn - attn_min) / (attn_max - attn_min + 1e-6)

    sim = sim.view(b, 1, h, w)
    attn = attn.view(b, 1, h, w)

    AN = attn >= attn_thr

    # print(density[0].max())
    # density = density/60
    # print(density[0].max())

    kernel = torch.ones(1, 1, 16, 16, device=sim.device, dtype=sim.dtype)
    gt = F.conv2d(density.unsqueeze(1), kernel, stride=16)

    P = gt >= den_thr

    # pos/neg: [b, 1, 24, 24]
    pos = (1-sim) * P
    neg = torch.clamp(sim, min=0) * (P == 0) * (AN == 0)

    pos = pos.flatten(1).sum(dim=1)
    neg = neg.flatten(1).sum(dim=1)

    pos_num = P.flatten(1).sum(dim=1)
    neg_num = ((AN == 0) * (P == 0)).flatten(1).sum(dim=1)

    loss = 2 * pos / (pos_num + 1e-6) + neg / (neg_num + 1e-6)

    print("pos:", (pos / (pos_num + 1e-6)).mean().item())
    print("neg:", (neg / (neg_num + 1e-6)).mean().item())

    return loss.mean()


# rcc损失改成log形式
def rrc_loss_log(
    sim,        # [B, 576] 相似度 *logits*（不要先 sigmoid）
    attn,       # [B, 576] cross-attn，用来构造 AN
    density,    # [B, 384, 384] GT 密度图
    attn_thr=0.3,
    den_thr=1e-3 * 60,
    pos_weight=2.0,   # 正样本权重（类似你原来的系数 2）
    eps=1e-6
):
    """
    对数形式的 RRC loss（BCEWithLogits 风格）：
      - 正样本：密度下采样后 >= den_thr 的 patch
      - 安全负样本：不是正样本 && attn < attn_thr
      - 模糊区域 (P=0 & attn >= thr) 不参与 loss
      - 对 sim 用 BCEWithLogits 做 0/1 监督
    """

    B, N = sim.shape
    h = w = int(sqrt(N))
    device, dtype = sim.device, sim.dtype

    # 1) 处理 attn（只用于 mask，不反传梯度）
    attn_det = attn.detach()                         # 不让 RRC 监督到注意力本身
    attn_max, _ = attn_det.max(dim=1, keepdim=True)
    attn_min, _ = attn_det.min(dim=1, keepdim=True)
    attn_norm = (attn_det - attn_min) / (attn_max - attn_min + eps)  # [0,1]
    attn_2d = attn_norm.view(B, 1, h, w)            # [B,1,24,24]

    # 2) 下采样 GT 得到正样本 mask P
    kernel = torch.ones(1, 1, 16, 16, device=device, dtype=dtype)
    gt_patch = F.conv2d(density.unsqueeze(1), kernel, stride=16)     # [B,1,24,24]
    P = (gt_patch >= den_thr)                                       # bool，[B,1,24,24]

    # 3) 构造模糊区 AN 和安全负样本
    AN = (attn_2d >= attn_thr)             # bool：注意力高的模糊区域

    pos_mask   = P                         # 正样本（有密度）
    neg_mask   = (~P) & (~AN)              # 安全负样本（无密度+注意力不高）
    valid_mask = pos_mask | neg_mask       # 参与 loss 的所有位置（模糊区排除）

    # 4) 构造 BCE 的 label 和权重
    # sim_logits: [B,1,24,24]
    sim_logits = sim.view(B, 1, h, w)

    # 正样本 label=1，负样本 label=0（模糊区后面用 weight=0 忽略）
    labels = pos_mask.float()              # [B,1,24,24]，正=1，其余先=0

    # 权重：正样本 pos_weight，负样本 1，模糊区 0
    weight = torch.ones_like(labels)
    weight[pos_mask]    = pos_weight       # 强调前景区域
    weight[~valid_mask] = 0.0              # 模糊区域：完全不参与

    # 5) 计算每个位置的 BCEWithLogits（不做 reduction）
    bce = F.binary_cross_entropy_with_logits(
        sim_logits, labels, reduction='none'
    )   # [B,1,24,24]

    # 掩掉无效区域，并加权前景
    loss_map = bce * weight                # [B,1,24,24]

    # 每张图做加权平均
    loss_sum   = loss_map.flatten(1).sum(dim=1)          # [B]
    weight_sum = weight.flatten(1).sum(dim=1).clamp_min(eps)  # [B]
    loss_per_img = loss_sum / weight_sum                 # [B]

    return loss_per_img.mean()



# rcc思想用infonce
def rrc_loss_infonce(
    sim,        # [B, 576]  相似度 *logits*（不要先 sigmoid）
    attn,       # [B, 576]  cross-attn，用于构造 AN（模糊区域）
    density,    # [B, 384, 384] GT 密度图
    attn_thr=1,
    den_thr=1e-3 * 60,
    tau=0.07,   # InfoNCE 温度
    eps=1e-6
):
    
    """
    BMNet 风格的多正样本 InfoNCE RRC 损失：

      对每张图 b：
        pos = P == 1           （密度 patch 内有足够 GT）
        neg = P == 0 & AN == 0 （安全背景，排除模糊区域）

        L_b = -log( sum_{pos} e^{z_i/tau} / (sum_{pos} e^{z_i/tau} + sum_{neg} e^{z_j/tau}) )

      其中 AN 由注意力决定：attn >= attn_thr 视为模糊区，不进 neg。
    """


    B, N = sim.shape
    h = w = int(sqrt(N))
    sim = sim.to(density.dtype)
    device, dtype = sim.device, sim.dtype

    # 1) 处理 attn（只用于构造 AN，不反向）
    attn_det = attn.detach()
    attn_max, _ = attn_det.max(dim=1, keepdim=True)
    attn_min, _ = attn_det.min(dim=1, keepdim=True)
    attn_norm = (attn_det - attn_min) / (attn_max - attn_min + eps)   # [B,N]

    attn_2d = attn_norm.view(B, 1, h, w)                              # [B,1,24,24]
    AN = (attn_2d >= attn_thr)                                        # bool
    # AN = ~P


    # 2) 用密度图构造 P（patch-level 正样本）
    kernel = torch.ones(1, 1, 16, 16, device=device, dtype=dtype)
    gt_patch = F.conv2d(density.unsqueeze(1), kernel, stride=16)      # [B,1,24,24]
    P = (gt_patch >= den_thr)                                        # bool
    
    # 3) 展平 mask：pos / 安全 neg / all
    pos_mask = P.view(B, -1)                                         # [B,N] bool
    # neg_mask = ((~P) & (~AN)).view(B, -1)                            # [B,N] bool
    neg_mask = (~P).view(B, -1)
    
    all_mask = pos_mask | neg_mask                                   # 参与对比的所有位置

    # 4) logits 加温度，做数值稳定处理
    # logits = torch.sigmoid(sim) 
    logits = sim                                            # [B,N]
    # logits = logits - logits.max(dim=1, keepdim=True)[0]             # 每图减去最大值防爆

    very_neg = torch.full_like(logits, -1e9)                         # 掩码用极小值

    # numerator: log Σ_{i in pos} exp(logits_i)
    logits_pos = torch.where(pos_mask, logits, very_neg)             # 非 pos 位置置为 -1e9
    log_num = torch.logsumexp(logits_pos, dim=1)                     # [B]

    # denominator: log Σ_{i in pos ∪ neg} exp(logits_i)
    logits_all = torch.where(all_mask, logits, very_neg)
    log_den = torch.logsumexp(logits_all, dim=1)                     # [B]

    # 5) InfoNCE：-log( num / den ) = -(log_num - log_den)
    loss_vec = -(log_num - log_den)                                  # [B]

    # 6) 对“没有正 / 没有负”的图跳过
    pos_count = pos_mask.sum(dim=1)
    neg_count = neg_mask.sum(dim=1)
    valid = (pos_count > 0) & (neg_count > 0)

    if valid.any():
        loss = loss_vec[valid].mean()
    else:
        loss = torch.tensor(0.0, device=device, dtype=dtype)

    return loss


def rrc_loss_infonce2(
    sim,        # [B, 576]  相似度 *logits*（不要先 sigmoid）
    attn,       # [B, 576]  cross-attn，用于构造 AN（模糊区域）
    density,    # [B, 384, 384] GT 密度图
    attn_thr=1,
    den_thr=1e-3 * 60,
    tau=0.07,   # InfoNCE 温度
    eps=1e-6
):

    """
    BMNet 风格的多正样本 InfoNCE RRC 损失：

      对每张图 b：
        pos = P == 1           （密度 patch 内有足够 GT）
        neg = P == 0 & AN == 0 （安全背景，排除模糊区域）

        L_b = -log( sum_{pos} e^{z_i/tau} / (sum_{pos} e^{z_i/tau} + sum_{neg} e^{z_j/tau}) )

      其中 AN 由注意力决定：attn >= attn_thr 视为模糊区，不进 neg。
    """


    B, N = sim.shape
    h = w = int(sqrt(N))
    device, dtype = sim.device, sim.dtype

    # 1) 处理 attn（只用于构造 AN，不反向）
    attn_det = attn.detach()
    attn_max, _ = attn_det.max(dim=1, keepdim=True)
    attn_min, _ = attn_det.min(dim=1, keepdim=True)
    attn_norm = (attn_det - attn_min) / (attn_max - attn_min + eps)   # [B,N]

    attn_2d = attn_norm.view(B, 1, h, w)                              # [B,1,24,24]
    AN = (attn_2d >= attn_thr)                                        # bool

    # 2) 用密度图构造 P（patch-level 正样本）
    kernel = torch.ones(1, 1, 16, 16, device=device, dtype=dtype)
    gt_patch = F.conv2d(density.unsqueeze(1), kernel, stride=16)      # [B,1,24,24]
    P = (gt_patch >= den_thr)                                        # bool

    # 3) 展平 mask：pos / 安全 neg / all
    pos_mask = P.view(B, -1)                                         # [B,N] bool
    neg_mask = ((~P) & (~AN)).view(B, -1)                            # [B,N] bool

    # neg_mask = (~P).view(B, -1)


    all_mask = pos_mask | neg_mask                                   # 参与对比的所有位置


    # 4) logits 加温度，做数值稳定处理
    # logits = torch.sigmoid(sim) 
    logits = sim                                            # [B,N]
    # logits = logits - logits.max(dim=1, keepdim=True)[0]             # 每图减去最大值防爆

    very_neg = torch.full_like(logits, -1e9)                         # 掩码用极小值

    # numerator: log Σ_{i in pos} exp(logits_i)
    logits_pos = torch.where(pos_mask, logits, very_neg)             # 非 pos 位置置为 -1e9
    log_num = torch.logsumexp(logits_pos, dim=1)                     # [B]

    # denominator: log Σ_{i in pos ∪ neg} exp(logits_i)
    logits_all = torch.where(all_mask, logits, very_neg)
    log_den = torch.logsumexp(logits_all, dim=1)                     # [B]

    # 5) InfoNCE：-log( num / den ) = -(log_num - log_den)
    loss_vec = -(log_num - log_den)                                  # [B]

    # 6) 对“没有正 / 没有负”的图跳过
    pos_count = pos_mask.sum(dim=1)
    neg_count = neg_mask.sum(dim=1)
    valid = (pos_count > 0) & (neg_count > 0)

    if valid.any():
        loss = loss_vec[valid].mean()
    else:
        loss = torch.tensor(0.0, device=device, dtype=dtype)

    return loss


# 考虑一个模糊区域
def rrc_lossV2(sim, attn, density, attn_thr=0.3, den_thr=1e-3*60):
    # sim: [B, 576]
    # attn: [B, 576]
    # density: [B, 384, 384]
    b, n = sim.shape
    h = w = int(sqrt(n))

    sim = torch.sigmoid(sim)
    attn_max = attn.max(dim=1, keepdim=True)[0]
    attn_min = attn.min(dim=1, keepdim=True)[0]
    attn = (attn - attn_min) / (attn_max - attn_min + 1e-6)

    sim = sim.view(b, 1, h, w)
    attn = attn.view(b, 1, h, w)

    AN = attn >= attn_thr

    # print(density[0].max())
    # density = density/60
    # print(density[0].max())

    kernel = torch.ones(1, 1, 16, 16, device=sim.device, dtype=sim.dtype)
    gt = F.conv2d(density.unsqueeze(1), kernel, stride=16)

    P = gt >= den_thr

    # pos/neg: [b, 1, 24, 24]
    pos = (1-sim) * P
    neg = torch.clamp(sim, min=0) * (P == 0) * (AN == 0)
    amb = (sim-0.5) * (P == 0) * (AN == 1)

    pos = pos.flatten(1).sum(dim=1)
    neg = neg.flatten(1).sum(dim=1)
    amb = amb.flatten(1).sum(dim=1)


    pos_num = P.flatten(1).sum(dim=1)
    neg_num = ((AN == 0) * (P == 0)).flatten(1).sum(dim=1)
    amb_num = ((AN == 1) * (P == 0)).flatten(1).sum(dim=1)

    loss = 2 * pos / (pos_num + 1e-6) + neg / (neg_num + 1e-6) + amb / (amb_num + 1e-6)

    return loss.mean()




if __name__ == "__main__":

    gt = torch.randn(2, 384, 384)
    density = torch.randn(2, 384, 384)
    
    local_count_loss = count_loss(density, gt)
    # rank_loss_value = rank_lossV3(gt, density)
    print(local_count_loss)