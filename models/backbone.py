<<<<<<< HEAD
import numpy as np
import skimage
import torch
import os
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


config_path = '/data0/ljh/jiasw/PriorNet/models/configs'
config_name = 'sam2_hiera_base_plus.yaml'
config_dir = os.path.abspath(config_path)

with initialize_config_dir(config_dir=config_dir, job_name='sam2'):
    cfg = compose(config_name=config_name)

# config_name = '../configs/sam2_hiera_base_plus.yaml'
# cfg = compose(config_name=config_name)
OmegaConf.resolve(cfg)
backbone = instantiate(cfg.backbone, _recursive_=True)

checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/segment_anything_2/072824/' + config_name.split('/')[-1].replace('.yaml', '.pt'), map_location="cpu")['model']

# 链接里加载出来的权重形式为: image_encoder.trunk.pos_embed, 但是实际则是trunk.pos_embed
state_dict = {k.replace("image_encoder.", ""): v for k, v in checkpoint.items()}
backbone.load_state_dict(state_dict, strict=False)
print("Load successfully!")


x = torch.randn(1, 3, 384, 384)

with torch.no_grad():
    # feats:
        # vision_features, [B, 256, 64, 64] len:1
        # vision_pos_enc: [B, 256, 256, 256]  [B, 256, 128, 128]  [B, 256, 64, 64] len:3
        # backbone_fpn: [B, 256, 256, 256]  [B, 256, 128, 128]  [B, 256, 64, 64] len:3
    feats = backbone(x)


vision_features = feats['vision_features']
print(vision_features.shape)

for i in feats['backbone_fpn']:
    print(i.shape)  

print("*****")
for i in feats['vision_pos_enc']:
    print(i.shape)
=======
import numpy as np
import skimage
import torch
import os
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


config_path = '/data0/ljh/jiasw/PriorNet/models/configs'
config_name = 'sam2_hiera_base_plus.yaml'
config_dir = os.path.abspath(config_path)

with initialize_config_dir(config_dir=config_dir, job_name='sam2'):
    cfg = compose(config_name=config_name)

# config_name = '../configs/sam2_hiera_base_plus.yaml'
# cfg = compose(config_name=config_name)
OmegaConf.resolve(cfg)
backbone = instantiate(cfg.backbone, _recursive_=True)

checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/segment_anything_2/072824/' + config_name.split('/')[-1].replace('.yaml', '.pt'), map_location="cpu")['model']

# 链接里加载出来的权重形式为: image_encoder.trunk.pos_embed, 但是实际则是trunk.pos_embed
state_dict = {k.replace("image_encoder.", ""): v for k, v in checkpoint.items()}
backbone.load_state_dict(state_dict, strict=False)
print("Load successfully!")


x = torch.randn(1, 3, 384, 384)

with torch.no_grad():
    # feats:
        # vision_features, [B, 256, 64, 64] len:1
        # vision_pos_enc: [B, 256, 256, 256]  [B, 256, 128, 128]  [B, 256, 64, 64] len:3
        # backbone_fpn: [B, 256, 256, 256]  [B, 256, 128, 128]  [B, 256, 64, 64] len:3
    feats = backbone(x)


vision_features = feats['vision_features']
print(vision_features.shape)

for i in feats['backbone_fpn']:
    print(i.shape)  

print("*****")
for i in feats['vision_pos_enc']:
    print(i.shape)
>>>>>>> 2884e4762d72b1a3d558d4adcc44b0f731340ded
