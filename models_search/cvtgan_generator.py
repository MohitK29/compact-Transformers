import torch
import torch.nn as nn
import math
import numpy as np


from models_search.cvT_helper import DropPath, to_2tuple, trunc_normal_
from models_search.diff_aug import DiffAugment
from src.utils.tokenizer import Tokenizer
from pdb import set_trace as TT
import torch.nn.functional as F

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        x = x1@x2
        return x

def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])

class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)

class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu
        
    def forward(self, x):
        return self.act_layer(x)
        
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x):
        B, N, C = x.shape
        x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength_1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)
        
    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0,2,1)).permute(0,2,1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class StageBlock(nn.Module):

    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth = depth
        self.block = nn.ModuleList([
                        Block(
                        dim=dim, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop, 
                        attn_drop=attn_drop, 
                        drop_path=drop_path, 
                        act_layer=act_layer,
                        norm_layer=norm_layer
                        ) for i in range(depth)])

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

class Generator(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=384, depth=5,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(Generator, self).__init__()

        self.args = args
        self.ch = embed_dim
        self.bottom_width = args.bottom_width
        self.embed_dim = embed_dim = args.gf_dim
        norm_layer = args.g_norm
        mlp_ratio = args.g_mlp
        depth = [int(i) for i in args.g_depth.split(",")]
        act_layer = args.g_act

        self.tokenizer = Tokenizer(n_input_channels=3,
                                   n_output_channels=embed_dim,
                                   kernel_size=3,
                                   stride=1,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.sequence_length = self.tokenizer.sequence_length(n_channels=3,            height=224,width=224)

        self.dim = embed_dim
        sequence_length = self.sequence_length
        self.seq_pool = True
        self.num_tokens = 0

        if not self.seq_pool:
            self.sequence_length += 1
            self.class_emb = torch.nn.Parameter(torch.zeros(1, 1, self.dim),requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = nn.Linear(embed_dim, 1)

        self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length//770, embed_dim),
                                                requires_grad=True)

        self.positional_emb2 = nn.Parameter(torch.zeros(1, 256, embed_dim),
                                                requires_grad=True)

        self.dropout = nn.Dropout(p=drop_rate)
        self.norm = CustomNorm(norm_layer,embed_dim)
        
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2, embed_dim//4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2, embed_dim//16))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3
        ]

        self.conv1 = nn.Conv2d(3,28,kernel_size = 3,stride = 1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule
        self.blocks = StageBlock(
                        depth=depth[0],
                        dim=embed_dim, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer
                        )
        self.upsample_blocks = nn.ModuleList([
                    StageBlock(
                        depth=depth[1],
                        dim=embed_dim//4, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer
                        ),
                    StageBlock(
                        depth=depth[2],
                        dim=embed_dim//16, 
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0,
                        act_layer=act_layer,
                        norm_layer=norm_layer
                        )
                    ])
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//16, 3, 1, 1, 0)
        )   
    def set_arch(self, x, cur_stage):
        pass

    def forward(self, z, epoch):
        if self.args.latent_norm:
            latent_size = z.size(-1)
            z = (z/z.norm(dim=-1, keepdim=True) * (latent_size ** 0.5))
        x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        x = x + self.positional_emb.to(x.get_device())

        B = x.size()
        H, W = self.bottom_width, self.bottom_width

        x = self.dropout(x)

        x = self.blocks(x)

        for index, blk in enumerate(self.upsample_blocks):
            x, H, W = pixel_upsample(x, H, W)
            x = x + self.positional_emb2.to(x.get_device())
            x = blk(x)
        TT()
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        output = self.deconv(x.permute(0, 2, 1).view(-1, self.embed_dim//16, H, W))
        return output
