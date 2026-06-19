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
        # Compress channels by 4 inside the PPM (matches UPerNet style)
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

class LocalPatchAttention(nn.Module):
    def __init__(self, q_dim: int, v_dim: int, use_checkpoint: bool = False, attn_drop: float = 0.1, proj_drop: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.v_proj = nn.Linear(v_dim, v_dim)
        self.channel_meanings = nn.Parameter(torch.randn(v_dim, q_dim) * 0.02)
        
        self.q_norm = nn.LayerNorm(q_dim)
        self.v_norm = nn.LayerNorm(v_dim)

        # ViT-style Attention Dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # 3x3 Spatial Convolution to smooth grid artifacts instead of 1x1 Linear
        self.out_proj = nn.Conv2d(v_dim, q_dim, kernel_size=3, padding=1)
        
        # ViT-style Projection Dropout
        self.proj_drop = nn.Dropout(proj_drop)

        # DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # FFN (MLP) component
        mlp_hidden_dim = q_dim * 4
        self.norm2 = nn.GroupNorm(1, q_dim) # Spatial Layer Normalization
        self.mlp = nn.Sequential(
            nn.Conv2d(q_dim, mlp_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Conv2d(mlp_hidden_dim, q_dim, kernel_size=1),
            nn.Dropout(proj_drop)
        )

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
        
        # --- ENFORCE FLOAT32 FOR ATTENTION STABILITY ---
        # Cast to float32 specifically for the matrix multiplication and sigmoid
        attn_logits = (Q.float() @ K.float().t()) * (Q.shape[-1] ** -0.5)
        attn_weights = torch.sigmoid(attn_logits)
        
        # Cast weights back to the original dtype (e.g., float16) to multiply with V
        attn_weights = attn_weights.to(V.dtype)
        # -----------------------------------------------
        
        # Apply ViT-style Attention Dropout
        attn_weights = self.attn_drop(attn_weights)
        
        x_attn = attn_weights * V
        
        # Reconstruct high-res spatial dimensions BEFORE projecting
        x_attn_spatial = x_attn.reshape(B, h, w, ph, pw, -1).permute(0, 5, 1, 3, 2, 4).reshape(B, -1, H, W)
        
        # Apply spatial smoothing
        out = self.out_proj(x_attn_spatial)

        # Apply ViT-style Projection Dropout
        out = self.proj_drop(out)

        # Residual connection with DropPath for attention
        x = q + self.drop_path(out)
        
        # FFN block with DropPath and residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward(self, q, v):
        if self.use_checkpoint and q.requires_grad:
            return checkpoint(self._forward_impl, q, v, use_reentrant=False)
        return self._forward_impl(q, v)

class HighResQueryExtractor(nn.Module):
    """
    Modernized, Memory-Efficient CNN for High-Res Queries.
    Uses a ConvNeXt-inspired inverted residual block with a massive 7x7 
    receptive field to capture rich spatial context with a fraction of the 
    parameters and memory footprint of standard convolutions.
    """
    def __init__(self, in_channels: int, cnn_dim: int, hidden_dim: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, cnn_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cnn_dim),
            nn.GELU()
        )
        
        # Large Kernel Depthwise Conv (7x7) for spatial context
        self.dwconv = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=7, padding=3, groups=cnn_dim, bias=False)
        self.norm = nn.BatchNorm2d(cnn_dim)
        
        # Inverted Bottleneck (Pointwise Expand -> Act -> Pointwise Project)
        self.pwconv1 = nn.Conv2d(cnn_dim, cnn_dim * 2, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(cnn_dim * 2, cnn_dim, kernel_size=1, bias=False)
        
        # LayerScale for stable residual gradient flow
        self.gamma = nn.Parameter(torch.ones(cnn_dim, 1, 1) * 1e-2)
        
        # Final projection to match attention hidden dimension (Linear, no activation)
        self.proj = nn.Conv2d(cnn_dim, hidden_dim, kernel_size=1)

    def _forward_impl(self, x):
        x = self.stem(x)
        
        # Inverted Residual Block computation
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        x = res + self.gamma * x
        
        return self.proj(x)

    def forward(self, x):
        # Solves High Memory Consumption for Stride-1
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

class LocalPatchRefiner(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        in_channels: int = 3,
        hidden_dim: int = 128,
        cnn_dim: int = 64,
        use_checkpoint: bool = True,
        use_ppm: bool = True,
        use_clustering: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        ppm_dropout: float = 0.2,
        cluster_patch_size: int = 4,
        cluster_target_k: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_ppm = use_ppm
        self.use_clustering = use_clustering
        self.cluster_patch_size = cluster_patch_size
        self.cluster_target_k = cluster_target_k
        
        # Replaced with the strong, memory-efficient extractor
        self.cnn = HighResQueryExtractor(
            in_channels=in_channels,
            cnn_dim=cnn_dim,
            hidden_dim=hidden_dim,
            use_checkpoint=use_checkpoint
        )
        
        # Conditional Positional Encoding
        self.cpe = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(in_channels_list))]

        self.stages = nn.ModuleList()
        for i, v_dim in enumerate(in_channels_list):
            self.stages.append(
                LocalPatchAttention(
                    q_dim=hidden_dim, 
                    v_dim=v_dim, 
                    use_checkpoint=use_checkpoint,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=dpr[i]
                )
            )
            
        # Initialize PPM for the deepest Swin feature map only when enabled
        self.ppm = PyramidPoolingModule(in_channels_list[-1]) if use_ppm else None

        # Spatial Dropout after PPM
        self.ppm_dropout = nn.Dropout2d(ppm_dropout) if use_ppm else nn.Identity()

    def _cluster_and_compact(self, q, patch_size=4, target_k=4):
        """
        Clusters high-res queries locally to reduce sequence length for attention.
        """
        B, C, H, W = q.shape
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size

        if pad_h or pad_w:
            # Make the feature map divisible by patch_size before reshaping.
            q = F.pad(q, (0, pad_w, 0, pad_h), mode='reflect')

        Hp, Wp = q.shape[2:]
        ph, pw = Hp // patch_size, Wp // patch_size
        
        # 1. Slice into local patches
        # (B, C, ph, patch_size, pw, patch_size) -> (B, ph*pw, patch_size*patch_size, C)
        q_patches = q.view(B, C, ph, patch_size, pw, patch_size).permute(0, 2, 4, 3, 5, 1).reshape(B, ph * pw, patch_size * patch_size, C)
        
        # 2. Fast Cosine Similarity Clustering (k-means++ style init)
        with torch.no_grad():
            # Normalize for cosine similarity
            q_norm = F.normalize(q_patches, p=2, dim=-1)
            
            # Select first cluster center randomly
            idx_0 = torch.randint(0, patch_size * patch_size, (B, ph * pw, 1), device=q.device)
            centers = torch.gather(q_norm, 2, idx_0.unsqueeze(-1).expand(-1, -1, -1, C))
            
            for _ in range(1, target_k):
                # Distances to current centers
                sim = torch.einsum('bnpc,bntc->bnpt', q_norm, centers) # (B, N, P, K)
                max_sim, _ = sim.max(dim=-1) # (B, N, P)
                
                # Pick furthest point as next center
                idx_next = max_sim.argmin(dim=-1, keepdim=True)
                next_center = torch.gather(q_norm, 2, idx_next.unsqueeze(-1).expand(-1, -1, -1, C))
                centers = torch.cat([centers, next_center], dim=2)
            
            # Final assignments
            sim = torch.einsum('bnpc,bntc->bnpt', q_norm, centers)
            assignments = sim.argmax(dim=-1) # (B, N, P) -> cluster index for each pixel
            
        # 3. Compact: Scatter Add (More memory efficient)
        N_patches = ph * pw
        P_dim = patch_size * patch_size
        
        compact_q = torch.zeros(B, N_patches, target_k, C, device=q.device, dtype=q.dtype)
        idx_features = assignments.unsqueeze(-1).expand(-1, -1, -1, C)
        compact_q.scatter_add_(2, idx_features, q_patches)
        
        counts = torch.zeros(B, N_patches, target_k, 1, device=q.device, dtype=q.dtype)
        ones = torch.ones(B, N_patches, P_dim, 1, device=q.device, dtype=q.dtype)
        counts.scatter_add_(2, assignments.unsqueeze(-1), ones)
        
        compact_q = compact_q / counts.clamp(min=1)
        
        # 4. Reshape back to spatial grid for the attention stages
        # (B, ph*pw, target_k, C) -> We pretend target_k is a small spatial grid (e.g. 2x2 if k=4)
        k_h = int(target_k**0.5)
        k_w = target_k // k_h
        compact_q_spatial = compact_q.reshape(B, ph, pw, k_h, k_w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, ph * k_h, pw * k_w)
        
        return compact_q_spatial, assignments, (patch_size, k_h, k_w, H, W, Hp, Wp)

    def _rebuild_original(self, compact_q_spatial, assignments, meta_info):
        """
        Un-groups the refined compact queries back to the original high-resolution grid.
        """
        patch_size, k_h, k_w, H, W, Hp, Wp = meta_info
        B, C, cH, cW = compact_q_spatial.shape
        ph, pw = Hp // patch_size, Wp // patch_size
        target_k = k_h * k_w
        
        # 1. Reshape refined compact queries back to list form
        # (B, C, ph*k_h, pw*k_w) -> (B, ph*pw, target_k, C)
        compact_q = compact_q_spatial.view(B, C, ph, k_h, pw, k_w).permute(0, 2, 4, 3, 5, 1).reshape(B, ph * pw, target_k, C)
        
        # 2. Broadcast back using assignments
        # assignments: (B, N, P) where P = patch_size*patch_size
        # We need to gather from (B, N, K, C) using (B, N, P, 1)
        expanded_assignments = assignments.unsqueeze(-1).expand(-1, -1, -1, C)
        rebuilt_q_patches = torch.gather(compact_q, 2, expanded_assignments) # (B, N, P, C)
        
        # 3. Reshape back to original image grid
        # (B, ph*pw, patch_size*patch_size, C) -> (B, C, Hp, Wp) -> crop to (H, W)
        rebuilt_q = rebuilt_q_patches.reshape(B, ph, pw, patch_size, patch_size, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        rebuilt_q = rebuilt_q[:, :, :H, :W]
        
        return rebuilt_q

    def forward(self, img, features: List[torch.Tensor]):
        q = self.cnn(img)
        q = q + self.cpe(q) # Re-enabled so clustering knows spatial positions

        if self.use_clustering:
            # Cluster & compact before cross-attention to reduce the high-res sequence length.
            q_compact, assignments, meta_info = self._cluster_and_compact(
                q,
                patch_size=self.cluster_patch_size,
                target_k=self.cluster_target_k,
            )
        else:
            q_compact = q
            assignments = None
            meta_info = None
        
        # Apply PPM to the deepest feature map (Stride 32) if enabled
        enhanced_features = list(features)
        if self.use_ppm:
            enhanced_features[-1] = self.ppm_dropout(self.ppm(enhanced_features[-1]))
        
        # Process Bottom-Up (Fine-to-Coarse) on the COMPACT grid
        # We process q_compact instead of q
        stage_outputs = []
        for i, (stage, f) in enumerate(zip(self.stages, enhanced_features)):
            
            h_f, w_f = f.shape[2:]
            h_q, w_q = q_compact.shape[2:]
            
            # Ensure Q perfectly matches F's grid scaled by relative patch ratio
            ratio_h = max(1, round(h_q / h_f))
            ratio_w = max(1, round(w_q / w_f))
            target_h, target_w = h_f * ratio_h, w_f * ratio_w
            
            pad_h = target_h - h_q
            pad_w = target_w - w_q
            
            q_stage = q_compact
            if pad_h > 0 or pad_w > 0:
                 q_stage = F.pad(q_stage, (0, max(0, pad_w), 0, max(0, pad_h)), mode='reflect')
            if pad_h < 0 or pad_w < 0:
                 q_stage = q_stage[:, :, :target_h, :target_w]
                 
            q_stage = stage(q_stage, f)
            
            if pad_h > 0 or pad_w > 0:
                 q_stage = q_stage[:, :, :h_q, :w_q]
            if pad_h < 0 or pad_w < 0:
                 q_stage = F.pad(q_stage, (0, max(0, -pad_w), 0, max(0, -pad_h)), mode='reflect')
                 
            q_compact = q_stage
            stage_outputs.append(q_stage)
            
        # Rebuild only when clustering was enabled.
        if self.use_clustering:
            q = self._rebuild_original(q_compact, assignments, meta_info)
        else:
            q = q_compact
            
        return q, stage_outputs


class MultiLevelSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int,
        channels: int,
        dropout_ratio: float = 0.2,
        align_corners: bool = False,
    ):
        super().__init__()
        self.align_corners = align_corners

        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], target_size=None):
        if len(features) != len(self.lateral_convs):
            raise ValueError(
                f'Expected {len(self.lateral_convs)} refined feature maps, got {len(features)}.'
            )
        if target_size is None:
            target_size = features[0].shape[2:]
        outs = []
        for lateral_conv, feat in zip(self.lateral_convs, features):
            x = lateral_conv(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                )
            outs.append(x)

        x = torch.cat(outs, dim=1)
        x = self.fuse_conv(x)
        x = self.dropout(x)
        return self.cls_seg(x)

class LPRHiDecoder(Decoder):
    def __init__(self, in_channels: List[int], num_classes: int, lpr_kwargs: dict):
        super().__init__()
        # Copy kwargs so config-driven overrides do not mutate shared dictionaries.
        lpr_kwargs = dict(lpr_kwargs)

        # Support both the project-specific name and the common decoder naming.
        spatial_dropout = lpr_kwargs.pop('spatial_dropout', lpr_kwargs.pop('dropout_ratio', 0.2))
        
        self.refiner = LocalPatchRefiner(in_channels_list=in_channels, **lpr_kwargs)
        
        hidden_dim = self.refiner.hidden_dim

        self.cls_seg = MultiLevelSegmentationHead(
            in_channels_list=[hidden_dim] * len(in_channels),
            num_classes=num_classes,
            channels=hidden_dim,
            dropout_ratio=spatial_dropout,
            align_corners=False,
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

    def forward(self, features: List[torch.Tensor], img: torch.Tensor):
        _final_refined, stage_features = self.refiner(img, features)
        out = self.cls_seg(stage_features, target_size=_final_refined.shape[2:])
        # return F.interpolate(out, size=img.shape[2:], mode='bilinear', align_corners=False)
        return out
