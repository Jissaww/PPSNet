from functools import partial
import os
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from math import sqrt
import cv2
from einops import rearrange,repeat
from timm.models.vision_transformer import PatchEmbed
from models.Block.Blocks import Block

import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np

class SupervisedMAE(nn.Module):
    """ CntVit with VisionTransformer backbone
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate = 0):
        super().__init__()
        ## Setting the model
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        print(depth, num_heads, decoder_depth, decoder_num_heads)

        ## Global Setting
        self.patch_size = patch_size
        self.img_size = img_size
        ex_size = 64
        self.norm_pix_loss = norm_pix_loss
        ## Global Setting

        """
            query、exemplar pos_embed、scale_embed 无法从预训练权重读到
            patch_embed、norm、blocks(12层encoder) 可以从权重中读取到(encoder部分)
            decoder_embed(维度映射)、decoder_norm、decoder_blocks(权重中有8层这里只用了3层) 可以从权重中读取到(decoder部分)
            decoder_head 无法从预训练权重中读到(密度回归部分)
        """

        ## Encoder specifics
        self.scale_embeds = nn.Linear(2, embed_dim, bias=True)

        self.patch_embed_exemplar = PatchEmbed(ex_size, patch_size, in_chans+1, embed_dim)
        # self.patch_embed_exemplar = PatchEmbed(ex_size, patch_size, in_chans, embed_dim)
        num_patches_exemplar = self.patch_embed_exemplar.num_patches
        self.pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, embed_dim), requires_grad=False)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False) 


        self.norm = norm_layer(embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])



        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding  
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        

        ## Regressor
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  
        


        # self.data = 0


        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_exemplar = get_2d_sincos_pos_embed(self.decoder_pos_embed_exemplar.shape[-1], int(self.patch_embed_exemplar.num_patches**.5), cls_token=False)
        self.decoder_pos_embed_exemplar.data.copy_(torch.from_numpy(decoder_pos_embed_exemplar).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w1 = self.patch_embed_exemplar.proj.weight.data
        torch.nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
       

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def espm_priorV4(self, scales, exemplar, ex_size=64, alpha=0.5, gamma=1.0):

        b, n, _ = scales.shape
        H = W = ex_size
        R0 = 0.5 * ex_size
        device = scales.device

        w_e = torch.clamp(scales[..., 0], min=1e-6)
        h_e = torch.clamp(scales[..., 1], min=1e-6)
        area = torch.clamp(w_e * h_e, min=1e-6)

        sigma_x = gamma * w_e * R0
        sigma_y = gamma * h_e * R0


        y = torch.arange(H, dtype=torch.float32, device=device)
        x = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        cx = torch.full((b, n, 1, 1), (W - 1) / 2.0, device=device)
        cy = torch.full((b, n, 1, 1), (H - 1) / 2.0, device=device)
        dx = xx.view(1, 1, H, W) - cx
        dy = yy.view(1, 1, H, W) - cy

        sig2_x = sigma_x.view(b, n, 1, 1) ** 2
        sig2_y = sigma_y.view(b, n, 1, 1) ** 2

        d2 = dx * dx / (2.0 * sig2_x +1e-6) + dy * dy / (2.0 * sig2_y + 1e-6)
        
        G = torch.exp(-d2)


        log_amp = (-alpha) * torch.log(area)
        amp = torch.exp(log_amp).view(b, n, 1, 1)

        P = (G * amp).unsqueeze(2)

        P = P.to(exemplar.device)
        exemplar = torch.cat([exemplar, P], dim=2)

        return exemplar, P


        b, n, _ = scales.shape
        H = W = ex_size
        R0 = 0.5 * ex_size
        device = scales.device

        w_e = torch.clamp(scales[..., 0], min=1e-6)
        h_e = torch.clamp(scales[..., 1], min=1e-6)
        area = torch.clamp(w_e * h_e, min=1e-6)

        sigma_x = gamma * w_e * R0
        sigma_y = gamma * h_e * R0


        y = torch.arange(H, dtype=torch.float32, device=device)
        x = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        cx = torch.full((b, n, 1, 1), (W - 1) / 2.0, device=device)
        cy = torch.full((b, n, 1, 1), (H - 1) / 2.0, device=device)
        dx = xx.view(1, 1, H, W) - cx
        dy = yy.view(1, 1, H, W) - cy

        sig2_x = sigma_x.view(b, n, 1, 1) ** 2
        sig2_y = sigma_y.view(b, n, 1, 1) ** 2

        d2 = dx * dx / (2.0 * sig2_x +1e-6) + dy * dy / (2.0 * sig2_y + 1e-6)
        
        G = torch.exp(-d2)

        # log_amp = (-alpha) * torch.log(area)W
        # amp = torch.exp(log_amp).view(b, n, 1, 1)
        # P = (G * amp).unsqueeze(2)

        P = G.unsqueeze(2)

        P = P.to(exemplar.device)
        exemplar = torch.cat([exemplar, P], dim=2)


        return exemplar, P
    

    def forward_encoder(self, x, y, scales=None):
        img = x

        y_embed = []
        y = rearrange(y,'b n c w h->n b c w h')

        for box in y:
            box = self.patch_embed_exemplar(box)
            # box: [B, 16, 768]
            box = box + self.pos_embed_exemplar
            y_embed.append(box)

        # y_embed: [3, B, 16, 768]
        y_embed = torch.stack(y_embed, dim=0)
        box_num,_,n,d = y_embed.shape
        
        # y: [B, 48, 768]
        y = rearrange(y_embed, 'box_num batch n d->batch (box_num  n) d')

        x = self.patch_embed(x)
        x = x + self.pos_embed
        b, l, d = x.shape
        attns = []

        x_y = torch.cat((x,y),axis=1)

        for i, blk in enumerate(self.blocks):
            x_y, attn = blk(x_y)
            attns.append(attn)

        x_y = self.norm(x_y)

        x = x_y[:,:l,:]

        # 取出交互后对应的边界框信息
        for i in range(box_num):
            y[:,i*n:(i+1)*n,:] = x_y[:,l+i*n:l+(i+1)*n,:]

        y = rearrange(y,'batch  (box_num  n) d->box_num batch n d',box_num = box_num,n=n)

        return x, y, attns

    
    def forward_decoder(self, img, x, y, scales=None):

        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed
        y_embeds = []

        b,l_x,d = x.shape
        num, batch, l, dim = y.shape

        for i in range(num):
            y_embed = self.decoder_embed(y[i])
            y_embed = y_embed + self.decoder_pos_embed_exemplar
            y_embeds.append(y_embed)

        y_embeds = torch.stack(y_embeds)

        num, batch, l, dim = y_embeds.shape

        # yv: [shot b l d]
        # yv = y_embeds
        y_embeds = rearrange(y_embeds,'n b l d -> b (n l) d')

        x = torch.cat((x, y_embeds), axis=1)
        _, N, _ = x.shape

        attns = []
        xs = []
        ys = []

        for i, blk in enumerate(self.decoder_blocks):
            x, attn = blk(x)
            if i == 2:
                x = self.decoder_norm(x)
            attns.append(attn)
            xs.append(x[:,:l_x,:])
            ys.append(x[:,l_x:,:])

        # # corr: [B, Nx]
        # corr = self.SimAfterDecoder(xs[-1], ys[-1])

        # # vis
        # c = corr.view(b, 1, 24, 24)
        # img = img[0].cpu().numpy().astype('float32').transpose(1, 2, 0)
        # c = F.interpolate(c, size=(384, 384), mode='bilinear', align_corners=False).detach().cpu().numpy().astype('float32')[0, 0]

        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].imshow(img)
        # im1 = ax[0].imshow(c, alpha=0.8)

        # ax[0].set_title("sim")
        # ax[0].axis('off')
        # fig.colorbar(im1, ax=ax[0])

        # ax[1].imshow(img)
        # ax[1].axis('off')

        # plt.tight_layout()
        # save_path = os.path.join('./vis/sim/Baseline_EspmV4_DecoderAfterSim/', f"step_{self.data}.jpg")
        # self.data = self.data + 1
        # fig.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.close(fig)

        return xs, ys, attns
    

    def forward_decoder_Simloss(self, img, x, y, im_id=None):

        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed
        y_embeds = []

        b,l_x,d = x.shape
        num, batch, l, dim = y.shape

        for i in range(num):
            y_embed = self.decoder_embed(y[i])
            y_embed = y_embed + self.decoder_pos_embed_exemplar
            y_embeds.append(y_embed)

        y_embeds = torch.stack(y_embeds)

        num, batch, l, dim = y_embeds.shape

        # yv: [shot b l d]
        # yv = y_embeds
        y_embeds = rearrange(y_embeds,'n b l d -> b (n l) d')

        x = torch.cat((x, y_embeds), axis=1)
        _, N, _ = x.shape

        attns = []
        xs = []
        ys = []

        for i, blk in enumerate(self.decoder_blocks):
            x, attn = blk(x)
            if i == 2:
                x = self.decoder_norm(x)
            attns.append(attn)
            xs.append(x[:,:l_x,:])
            ys.append(x[:,l_x:,:])

        # corr: [B, Nx]
        
        # directly concat cos-sim
        corr = self.SimAfterDecoder(xs[-1], ys[-1])

        # attnlast: [B, num_heads, N, N]
        attnlast = attns[-1]
        attnlast = attnlast.mean(dim=1)[:, :l_x, :l_x]
        attnlast = attnlast.mean(dim=1)
        attnlast = attnlast.view(-1, 24, 24)

        attnself = attns[-1]
        attnself = attnself.mean(dim=1)[:, l_x:, l_x:]
        # print(attnself.shape, "***")
        # b, n, ly, 48
        attnself = rearrange(attnself, 'b (n ly) lll->b n ly lll', n=num, lll=48)
        attnself = attnself.mean(1)
        attnself = attnself.mean(dim=-1)      
        attnself = attnself.view(-1, 4, 4)  

        xx = xs[-1]

        # 相似图加权
        gate_corr = torch.sigmoid(corr).unsqueeze(-1)
        xx = xx * (1+gate_corr)


        h = w = int(sqrt(N))
        xx = rearrange(xx, 'b (h w) d->b d h w', h=h)


        return xs, ys, attns, corr, xx, attnself



    def SimAfterDecoder(self, x, y):
        # x: [B, Nx, D]
        # y: [B, Ny, D]
        
        b, Nx, d = x.shape
        _, Ny, _ = y.shape

        num_shot = Ny // 16

        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        corr = torch.einsum('bld, bmd->blm', x, y)

        corr = rearrange(corr, 'b nx (n ly)->b nx ly n', ly=16)


        # corr: [B, Nx, num_shot]
        corr = corr.sum(2)
        # corr:[B, Nx]
        corr = corr.mean(dim=-1)
        
        return corr
    

    def Regressor(self, feature):
        feature = F.interpolate(
                                self.decode_head0(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head1(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head2(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
                                self.decode_head3(feature), size=feature.shape[-1]*2, mode='bilinear', align_corners=False)
        feature = feature.squeeze(-3)

        return feature


    def forward(self, samples, name=None):
        # scales: [B, num_shot, 2]
        imgs = samples[0]
        boxes = samples[1]
        scales = samples[2]

        im_id = samples[3]
 

        boxes, prior = self.espm_priorV4(scales, boxes)

       
        latent, y_latent, enattns = self.forward_encoder(imgs, boxes, scales=scales)
   

        xs, ys, deattns, corr, reg_feat, attnl = self.forward_decoder_Simloss(imgs, latent, y_latent, im_id)
        
        density_map = self.Regressor(reg_feat)

     

        return density_map, attnl, corr   



def mae_vit_base_patch16_dec512d8b(**kwargs):

    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks