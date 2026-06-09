import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List

from ...base import Decoder

# --- DropPath Implementation ---
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
# -------------------------------

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.pool_scales = pool_scales
        channels = in_channels // 4
        
        self.psp_modules = nn.ModuleList()
        for pool_scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        psp_outs = [x]
        for psp_mod in self.psp_modules:
            psp_out = psp_mod(x)
            psp_outs.append(F.interpolate(psp_out, size=x.shape[2:], mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(psp_outs, dim=1))


class TopDownCoverageAttention(nn.Module):
    """
    Cross-Attention module where 1 Coarse Query Vector dynamic-gates its corresponding 
    coverage patch (e.g., 4 Finer Value Vectors) using a learnable Key matrix.
    """
    def __init__(self, q_dim: int, v_dim: int, use_checkpoint: bool = False, attn_drop: float = 0.1, proj_drop: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.v_proj = nn.Linear(v_dim, v_dim)
        
        # Learnable Key matrix (Channel Meanings)
        self.channel_meanings = nn.Parameter(torch.randn(v_dim, q_dim) * 0.02)
        
        self.q_norm = nn.LayerNorm(q_dim)
        self.v_norm = nn.LayerNorm(v_dim)

        self.attn_drop = nn.Dropout(attn_drop)

        # 3x3 Spatial Convolution to project back to q_dim and smooth artifacts
        self.out_proj = nn.Conv2d(v_dim, q_dim, kernel_size=3, padding=1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # FFN (MLP) component
        mlp_hidden_dim = q_dim * 4
        self.norm2 = nn.GroupNorm(1, q_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(q_dim, mlp_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Conv2d(mlp_hidden_dim, q_dim, kernel_size=1),
            nn.Dropout(proj_drop)
        )

    def _forward_impl(self, q, v):
        # q (Coarse): e.g., Level 4 (B, C_q, 8, 8)  -> 64 Vectors
        # v (Fine):   e.g., Level 3 (B, C_v, 16, 16) -> 256 Vectors
        B, C_q, H, W = q.shape
        _, C_v, h, w = v.shape
        
        # Calculate coverage ratio (usually 2 for standard downsampling)
        ratio_h = max(1, round(h / H))
        ratio_w = max(1, round(w / W))
        
        target_h, target_w = H * ratio_h, W * ratio_w
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Pad V if dimensions are not perfectly divisible
        v_padded = v
        if pad_h > 0 or pad_w > 0:
            v_padded = F.pad(v, (0, pad_w, 0, pad_h), mode='reflect')
            
        ph, pw = ratio_h, ratio_w  # e.g., ph=2, pw=2 (4 vectors total)
        
        # 1. Reshape Q: 1 vector per spatial location -> (B, H, W, 1, C_q)
        q_reshaped = q.permute(0, 2, 3, 1).unsqueeze(3)
        
        # 2. Reshape V: Group into coverage patches matching Q's grid -> (B, H, W, 4, C_v)
        v_reshaped = v_padded.view(B, C_v, H, ph, W, pw).permute(0, 2, 4, 3, 5, 1).reshape(B, H, W, ph * pw, C_v)
        
        q_normed = self.q_norm(q_reshaped)
        v_normed = self.v_norm(v_reshaped)
        
        Q = self.q_proj(q_normed)
        V = self.v_proj(v_normed)
        K = self.channel_meanings
        
        # --- ATTENTION MECHANISM ---
        # Q: (B, H, W, 1, C_q) @ K.T: (C_q, C_v) => attn_logits: (B, H, W, 1, C_v)
        # The 1 coarse query dynamically determines the weights for the C_v channels
        attn_logits = (Q.float() @ K.float().t()) * (Q.shape[-1] ** -0.5)
        attn_weights = torch.sigmoid(attn_logits)
        attn_weights = attn_weights.to(V.dtype)
        attn_weights = self.attn_drop(attn_weights)
        
        # Multiply weights by the 4 local vectors
        # (B, H, W, 1, C_v) * (B, H, W, 4, C_v) => (B, H, W, 4, C_v)
        x_attn = attn_weights * V
        # ---------------------------
        
        # 3. Reconstruct into the Finer spatial grid (B, C_v, target_h, target_w)
        x_attn_spatial = x_attn.reshape(B, H, W, ph, pw, C_v).permute(0, 5, 1, 3, 2, 4).reshape(B, C_v, target_h, target_w)
        
        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            x_attn_spatial = x_attn_spatial[:, :, :h, :w]
            
        # 4. Project refined V back to C_q so it can cascade as the next Q
        out = self.out_proj(x_attn_spatial)
        out = self.proj_drop(out)
        
        # 5. Residual Connection (Requires upsampling Q to match the new finer grid)
        q_up = F.interpolate(q, size=(h, w), mode='bilinear', align_corners=False)
        x = q_up + self.drop_path(out)
        
        # 6. FFN Block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

    def forward(self, q, v):
        if self.use_checkpoint and q.requires_grad:
            return checkpoint(self._forward_impl, q, v, use_reentrant=False)
        return self._forward_impl(q, v)


class TopDownPatchRefiner(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        hidden_dim: int = 128,
        use_checkpoint: bool = True,
        use_ppm: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        ppm_dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_ppm = use_ppm
        
        deepest_dim = in_channels_list[-1]
        
        # Base processing for the deepest feature level
        self.ppm = PyramidPoolingModule(deepest_dim) if use_ppm else nn.Identity()
        self.ppm_dropout = nn.Dropout2d(ppm_dropout) if use_ppm else nn.Identity()

        self.init_proj = nn.Sequential(
            nn.Conv2d(deepest_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # Top-Down Cascading Stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(in_channels_list) - 1)]
        self.stages = nn.ModuleList()
        
        for i in range(len(in_channels_list) - 1):
            shallow_dim = in_channels_list[-(i + 2)]
            self.stages.append(
                TopDownCoverageAttention(
                    q_dim=hidden_dim, 
                    v_dim=shallow_dim, 
                    use_checkpoint=use_checkpoint,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=dpr[i]
                )
            )

    def forward(self, features: List[torch.Tensor]):
        # 1. Start from the highest semantic level / lowest resolution patch vector (Level 4)
        x = features[-1]
        
        if self.use_ppm:
            x = self.ppm(x)
        x = self.ppm_dropout(x)
        
        # Project initial coarse feature to hidden_dim (C_q)
        x = self.init_proj(x)

        # 2. Iteratively cascade top-down, dynamically gating finer corresponding patches
        for i, stage in enumerate(self.stages):
            shallow_feat = features[-(i + 2)]
            x = stage(x, shallow_feat)
            
        return x


class TopDownDecoder(Decoder):
    def __init__(self, in_channels: List[int], num_classes: int, decoder_kwargs: dict):
        super().__init__()
        decoder_kwargs = dict(decoder_kwargs)

        spatial_dropout = decoder_kwargs.pop('spatial_dropout', decoder_kwargs.pop('dropout_ratio', 0.2))
        
        # Uses the true Coverage Attention cascade
        self.refiner = TopDownPatchRefiner(in_channels_list=in_channels, **decoder_kwargs)
        
        hidden_dim = self.refiner.hidden_dim
        
        self.cls_seg = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(spatial_dropout),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor], img: torch.Tensor = None):
        refined_features = self.refiner(features)
        out = self.cls_seg(refined_features)
        return out