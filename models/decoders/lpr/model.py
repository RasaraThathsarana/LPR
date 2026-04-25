import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from torch.cuda.amp import autocast

class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        mid_channels = out_channels // self.expansion

        # 1×1 reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(8, mid_channels)

        # 3×3 spatial conv
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.GroupNorm(8, mid_channels)

        # 1×1 expansion
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(8, out_channels)

        # projection if shape mismatch
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )

    def _forward_impl(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out += identity
        return F.relu(out, inplace=True)

    def forward(self, x):
        return self._forward_impl(x)

class UNet_FullRes(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Encoder
        self.enc1 = BasicBlock(in_channels, base_channels, use_checkpoint=use_checkpoint)
        self.enc2 = BasicBlock(base_channels, base_channels * 2, stride=2, use_checkpoint=use_checkpoint)
        self.enc3 = BasicBlock(base_channels * 2, base_channels * 4, stride=2, use_checkpoint=use_checkpoint)
        self.enc4 = BasicBlock(base_channels * 4, base_channels * 8, stride=2, use_checkpoint=use_checkpoint)
        
        # Bottleneck
        self.bottleneck = BasicBlock(base_channels * 8, base_channels * 16, stride=2, use_checkpoint=use_checkpoint)
        
        # Decoder
        self.dec4 = BasicBlock(base_channels * 16 + base_channels * 8, base_channels * 8, use_checkpoint=use_checkpoint)
        
        self.dec3 = BasicBlock(base_channels * 8 + base_channels * 4, base_channels * 4, use_checkpoint=use_checkpoint)
        
        self.dec2 = BasicBlock(base_channels * 4 + base_channels * 2, base_channels * 4, use_checkpoint=use_checkpoint)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        
        up4 = F.interpolate(b, size=s4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([up4, s4], dim=1))
        
        up3 = F.interpolate(d4, size=s3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([up3, s3], dim=1))
        
        up2 = F.interpolate(d3, size=s2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([up2, s2], dim=1))
        
        final_up = F.interpolate(d2, size=s1.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([final_up, s1], dim=1)
        
        return out

class LocalPatchRefiner(nn.Module):
    def __init__(self, global_dim, in_channels=3, patch_size=16, hidden_dim=256, cnn_dim=32, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim

        self.cnn = UNet_FullRes(in_channels=in_channels, base_channels=cnn_dim, use_checkpoint=use_checkpoint)

        # Dimension calculation for Query: out(cnn_dim*4)
        combined_dim = cnn_dim * 5

        self.channel_meanings = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(global_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(global_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 2D Sine-Cosine Positional Encoding for spatial awareness within the patch
        pos_embed = self._get_2d_sincos_pos_embed(hidden_dim, patch_size)
        self.register_buffer('pos_embed', pos_embed)

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        # embed_dim must be even
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        pos_embed = torch.cat([emb_h, emb_w], dim=1) # (grid_size*grid_size, embed_dim)
        return pos_embed.unsqueeze(0) # (1, grid_size*grid_size, embed_dim)

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000**omega)
        
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        
        return torch.cat([emb_sin, emb_cos], dim=1)

    def _attention_block(self, q_normed, k_normed, v_normed):
    # Force fp32 for attention computation to maintain numerical stability
        with torch.amp.autocast('cuda', enabled=False):

            q_normed = q_normed.float()
            k_normed = k_normed.float()
            v_normed = v_normed.float()
            
            Q = self.q_proj(q_normed)
            K = self.k_proj(k_normed)
            V = self.v_proj(v_normed)

            attn_logits = (Q @ K.t()) * (self.hidden_dim ** -0.5)
            attn_weights = torch.tanh(attn_logits)

            x_attn = attn_weights * V

            out = self.out_proj(x_attn)

        return out

    def forward(self, img, global_tokens):
        B, C, H, W = img.shape
        P = self.patch_size
        
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            
        H_pad, W_pad = img.shape[2:]
        
        all_feats = self.cnn(img)

        n_h, n_w = H_pad // P, W_pad // P

        # Robust check to ensure global_tokens match spatial dims of img//P
        if global_tokens.shape[-2] != n_h or global_tokens.shape[-1] != n_w:
            global_tokens = F.interpolate(global_tokens, size=(n_h, n_w), mode='bilinear', align_corners=False)

        global_tokens = global_tokens.permute(0, 2, 3, 1) # [B, n_h, n_w, global_dim]

        # Patching for Query construction
        q_map = all_feats.view(B, all_feats.size(1), n_h, P, n_w, P).permute(0, 2, 4, 1, 3, 5).reshape(-1, all_feats.size(1), P, P)
            
        q_tokens = q_map.flatten(2).transpose(1, 2)
        # Normalize after adding position information so attention sees
        # the intended zero-mean, unit-variance query distribution.
        q_normed = self.q_norm(q_tokens + self.pos_embed)

        k_normed = self.k_norm(self.channel_meanings)
        v_normed = self.v_norm(global_tokens).reshape(B * n_h * n_w, 1, self.global_dim)

        if self.use_checkpoint and q_normed.requires_grad:
            attn_features = checkpoint(self._attention_block, q_normed, k_normed, v_normed, use_reentrant=False)
        else:
            attn_features = self._attention_block(q_normed, k_normed, v_normed)
        
        # Prepare components for Fusion (only attention features now)
        fusion_input = attn_features.transpose(1, 2).reshape(-1, self.hidden_dim, P, P)

        out = fusion_input.view(B, n_h, n_w, self.hidden_dim, P, P)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, self.hidden_dim, H_pad, W_pad)
        
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        
        return out

from ...base import Decoder
from typing import List

class LPRDecoder(Decoder):
    """
    A decoder that wraps the LocalPatchRefiner to make it compatible with the
    EncoderDecoderModel framework.

    NOTE: This decoder has a special requirement. It needs both the feature maps from
    the encoder AND the original input image. The `EncoderDecoderModel.forward` method
    must be modified to pass the original image tensor `x` to this decoder's
    forward method as the `img` argument.
    """
    def __init__(self, in_channels: List[int], num_classes: int, lpr_kwargs: dict):
        super().__init__()
        # The 'global_dim' is the channel dimension of the feature map from the encoder.
        # We will use the last feature map from the Swin encoder.
        global_dim = in_channels[-1]

        self.refiner = LocalPatchRefiner(global_dim=global_dim, **lpr_kwargs)

        # The output of the refiner has `hidden_dim` channels.
        # A final segmentation head maps this to `num_classes`.
        hidden_dim = self.refiner.hidden_dim
        self.cls_seg = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], img: torch.Tensor):
        global_tokens = features[-1]
        refined_features = self.refiner(img, global_tokens)
        return self.cls_seg(refined_features)