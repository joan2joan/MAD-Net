from tkinter import W
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MAAttn(nn.Module):
    def __init__(self, dim, memory_blocks=128, squeeze_factor=16, proj_drop=0.):
        super(MAAttn, self).__init__()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            nn.Linear(dim, dim // squeeze_factor),
           )
        self.upnet= nn.Sequential(
            nn.Linear(dim // squeeze_factor, dim),
            nn.Sigmoid())
        self.mb = torch.nn.Parameter(torch.randn(dim // squeeze_factor, memory_blocks))
        self.low_dim = dim // squeeze_factor

    def forward(self, x):
        b,n,c = x.shape
        t = x.transpose(1,2)
        y = self.pool(t).squeeze(-1)
        low_rank_f = self.subnet(y).unsqueeze(2)
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1,2) ) @mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1) # get the similarity information
        y1 = f_dic_c@mbg.transpose(1,2)
        y2 = self.upnet(y1)
        x3 = x * y2
        out = self.proj(x3)
        out = self.proj_drop(out)
        return out

class mema2m(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0, drop_path=0.0, weight_factor=0.1, memory_blocks=128, down_rank=16,
                 mlp_ratio=4., drop=0., act_layer=nn.GELU):
        super(mema2m, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.weight_factor=weight_factor
        self.norm1 =nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attns = MAAttn(dim, memory_blocks=memory_blocks, squeeze_factor=down_rank, proj_drop=drop)
        self.num_heads = num_heads


    def forward(self, x):
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attns(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

