import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List

class LocalPatchAttention(nn.Module):
    def __init__(self, q_dim: int, v_dim: int, inner_dim: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.q_proj = nn.Linear(q_dim, inner_dim)
        self.v_proj = nn.Linear(v_dim, inner_dim)
        self.channel_meanings = nn.Parameter(torch.randn(inner_dim, inner_dim) * 0.02)
        self.out_proj = nn.Linear(inner_dim, q_dim)
        
        self.q_norm = nn.LayerNorm(q_dim)
        self.v_norm = nn.LayerNorm(v_dim)

    def _forward_impl(self, q, v):
        # q: (B, C_q, H, W) -> high res (Stride 4)
        # v: (B, C_v, h, w) -> multi-scale low res (Stride 4, 8, 16, 32)
        B, C_q, H, W = q.shape
        _, C_v, h, w = v.shape
        
        ph, pw = H // h, W // w
        
        # Reshape q into patches perfectly matching v's spatial dimensions
        # (B, C_q, h, ph, w, pw) -> (B, h, w, ph*pw, C_q)
        q_reshaped = q.reshape(B, C_q, h, ph, w, pw).permute(0, 2, 4, 3, 5, 1).reshape(B, h, w, ph * pw, C_q)
        
        # (B, C_v, h, w) -> (B, h, w, 1, C_v)
        v_reshaped = v.permute(0, 2, 3, 1).unsqueeze(3)
        
        q_normed = self.q_norm(q_reshaped)
        v_normed = self.v_norm(v_reshaped)
        
        Q = self.q_proj(q_normed)
        V = self.v_proj(v_normed)
        K = self.channel_meanings
        
        # Spatial-channel gating (spatial pixels gate matching V patch)
        attn_logits = (Q @ K.t()) * (Q.shape[-1] ** -0.5)
        attn_weights = torch.sigmoid(attn_logits)
        
        x_attn = attn_weights * V
        
        out_reshaped = self.out_proj(x_attn)
        
        # Reconstruct high-res spatial dimensions
        out = out_reshaped.reshape(B, h, w, ph, pw, C_q).permute(0, 5, 1, 3, 2, 4).reshape(B, C_q, H, W)
        
        return q + out

    def forward(self, q, v):
        if self.use_checkpoint and q.requires_grad:
            return checkpoint(self._forward_impl, q, v, use_reentrant=False)
        return self._forward_impl(q, v)

class LocalPatchRefiner(nn.Module):
    def __init__(self, in_channels_list: List[int], in_channels: int = 3, hidden_dim: int = 128, cnn_dim: int = 64, use_checkpoint: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Extremely lightweight CNN to extract full-resolution (Stride-1) queries
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_dim, cnn_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_dim * 2, hidden_dim, kernel_size=1)
        )
        
        # Conditional Positional Encoding
        self.cpe = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.stages = nn.ModuleList()
        for v_dim in in_channels_list:
            self.stages.append(
                LocalPatchAttention(
                    q_dim=hidden_dim, 
                    v_dim=v_dim, 
                    inner_dim=hidden_dim, 
                    use_checkpoint=use_checkpoint
                )
            )

    def forward(self, img, features: List[torch.Tensor]):
        q = self.cnn(img)
        q = q + self.cpe(q)
        
        for i, (stage, f) in enumerate(zip(self.stages, features)):
            h_f, w_f = f.shape[2:]
            h_q, w_q = q.shape[2:]
            
            # Ensure Q perfectly matches F's grid scaled by relative patch ratio
            # Swin features have strides 4, 8, 16, 32. Q is stride 1.
            # Therefore, the resolution ratio is 4 * (2 ** i)
            ratio = 4 * (2 ** i)
            target_h, target_w = h_f * ratio, w_f * ratio
            
            pad_h = target_h - h_q
            pad_w = target_w - w_q
            
            q_stage = q
            if pad_h > 0 or pad_w > 0:
                 q_stage = F.pad(q_stage, (0, max(0, pad_w), 0, max(0, pad_h)), mode='reflect')
            if pad_h < 0 or pad_w < 0:
                 q_stage = q_stage[:, :, :target_h, :target_w]
                 
            q_stage = stage(q_stage, f)
            
            if pad_h > 0 or pad_w > 0:
                 q_stage = q_stage[:, :, :h_q, :w_q]
            if pad_h < 0 or pad_w < 0:
                 q_stage = F.pad(q_stage, (0, max(0, -pad_w), 0, max(0, -pad_h)), mode='reflect')
                 
            q = q_stage
            
        return q

from ...base import Decoder

class LPRDecoder(Decoder):
    def __init__(self, in_channels: List[int], num_classes: int, lpr_kwargs: dict):
        super().__init__()
        self.refiner = LocalPatchRefiner(in_channels_list=in_channels, **lpr_kwargs)
        
        hidden_dim = self.refiner.hidden_dim
        
        self.cls_seg = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

    def forward(self, features: List[torch.Tensor], img: torch.Tensor):
        refined_features = self.refiner(img, features)
        out = self.cls_seg(refined_features)
        return F.interpolate(out, size=img.shape[2:], mode='bilinear', align_corners=False)