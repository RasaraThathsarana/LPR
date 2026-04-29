"""MMSegmentation-style UNet decoder implementation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from ...base import Decoder


def _make_norm(norm_cfg: dict | None, num_features: int) -> nn.Module:
    """Build a normalization layer with an MMSeg-friendly default."""
    norm_cfg = norm_cfg or {'type': 'BN'}
    norm_type = str(norm_cfg.get('type', 'BN')).lower()

    if norm_type in {'bn', 'batchnorm', 'batchnorm2d', 'syncbn'}:
        return nn.BatchNorm2d(num_features)
    if norm_type in {'gn', 'groupnorm'}:
        num_groups = int(norm_cfg.get('num_groups', 32))
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    raise ValueError(f'Unsupported norm type: {norm_cfg.get("type")}')


def _make_activation(act_cfg: dict | None) -> nn.Module:
    """Build an activation layer."""
    act_cfg = act_cfg or {'type': 'ReLU'}
    act_type = str(act_cfg.get('type', 'ReLU')).lower()
    inplace = bool(act_cfg.get('inplace', True))

    if act_type == 'relu':
        return nn.ReLU(inplace=inplace)
    if act_type == 'leakyrelu':
        negative_slope = float(act_cfg.get('negative_slope', 0.01))
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    raise ValueError(f'Unsupported activation type: {act_cfg.get("type")}')


class BasicConvBlock(nn.Module):
    """A basic conv-norm-activation block used in decoder stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride: int = 1,
        dilation: int = 1,
        with_cp: bool = False,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.layers = nn.Sequential()

        num_convs = max(1, int(num_convs))
        blocks = []
        for idx in range(num_convs):
            blocks.append(
                nn.Conv2d(
                    in_channels if idx == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride if idx == 0 else 1,
                    dilation=1 if idx == 0 else dilation,
                    padding=1 if idx == 0 else dilation,
                    bias=False,
                )
            )
            blocks.append(_make_norm(norm_cfg, out_channels))
            blocks.append(_make_activation(act_cfg))

        self.layers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_cp and x.requires_grad:
            return cp.checkpoint(self.layers, x)
        return self.layers(x)


class DeconvModule(nn.Module):
    """Deconvolution upsample module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 4,
        scale_factor: int = 2,
        with_cp: bool = False,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ):
        super().__init__()
        self.with_cp = with_cp
        padding = (kernel_size - scale_factor) // 2
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=scale_factor,
                padding=padding,
                bias=False,
            ),
            _make_norm(norm_cfg, out_channels),
            _make_activation(act_cfg),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_cp and x.requires_grad:
            return cp.checkpoint(self.layers, x)
        return self.layers(x)


class InterpConv(nn.Module):
    """Interpolation upsample module followed by a convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        scale_factor: int = 2,
        mode: str = 'bilinear',
        align_corners: bool = False,
        conv_first: bool = False,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        with_cp: bool = False,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.conv_first = conv_first
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode=mode,
                align_corners=align_corners if mode in {'bilinear', 'bicubic'} else None,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            _make_norm(norm_cfg, out_channels),
            _make_activation(act_cfg),
        )
        self.layers = nn.Sequential(self.conv, self.upsample) if conv_first else nn.Sequential(self.upsample, self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_cp and x.requires_grad:
            return cp.checkpoint(self.layers, x)
        return self.layers(x)


class UpConvBlock(nn.Module):
    """Decoder stage that upsamples, fuses skip features, and refines them."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_convs: int = 2,
        dilation: int = 1,
        with_cp: bool = False,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        upsample_cfg: dict | None = None,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.skip_channels = skip_channels
        self.out_channels = out_channels

        upsample_cfg = dict(upsample_cfg or {"type": "InterpConv"})
        upsample_type = str(upsample_cfg.pop("type", "InterpConv")).lower()
        upsample_kwargs = dict(upsample_cfg)

        if upsample_type in {'interpconv', 'interp'}:
            self.upsample = InterpConv(
                in_channels,
                out_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **upsample_kwargs,
            )
        elif upsample_type in {'deconvmodule', 'deconv', 'deconvconv'}:
            self.upsample = DeconvModule(
                in_channels,
                out_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **upsample_kwargs,
            )
        else:
            raise ValueError(f'Unsupported upsample type: {upsample_type}')

        self.refine = BasicConvBlock(
            out_channels + skip_channels,
            out_channels,
            num_convs=num_convs,
            dilation=dilation,
            with_cp=with_cp,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class UNetDecoder(Decoder):
    """UNet-style decoder with stage-wise upsampling and skip fusion.

    The structure mirrors MMSegmentation's UNet decoder path as closely as
    possible in this project:
    - each decoder stage upsamples by 2x
    - each stage fuses the matching encoder skip feature
    - each fused feature is refined with a small conv block
    - the final feature map is projected to classes with a 1x1 conv
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int = 150,
        decoder_channels: Sequence[int] | None = None,
        channels: int | None = None,
        num_convs: int = 2,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        output_scale: int = 4,
        with_cp: bool = False,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        upsample_cfg: dict | None = None,
    ):
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError('UNetDecoder expects at least two feature maps.')

        self.in_channels = list(in_channels)
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.output_scale = output_scale

        if decoder_channels is None:
            # Keep legacy configs working, but still prefer the stage-wise
            # channel pyramid that matches MMSegmentation's UNet layout.
            decoder_channels = list(reversed(self.in_channels[:-1]))
        else:
            decoder_channels = list(decoder_channels)

        if len(decoder_channels) != len(self.in_channels) - 1:
            raise ValueError(
                'decoder_channels must have one entry per skip connection '
                f'(expected {len(self.in_channels) - 1}, got {len(decoder_channels)})'
            )

        self.decoder_channels = decoder_channels
        self.stages = nn.ModuleList()

        current_channels = self.in_channels[-1]
        for skip_channels, out_channels in zip(reversed(self.in_channels[:-1]), decoder_channels):
            self.stages.append(
                UpConvBlock(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    num_convs=num_convs,
                    with_cp=with_cp,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg,
                )
            )
            current_channels = out_channels

        self.cls_seg = nn.Conv2d(current_channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.in_channels):
            raise ValueError(
                f'UNetDecoder expected {len(self.in_channels)} feature maps, got {len(features)}.'
            )

        x = features[-1]
        for stage_idx, stage in enumerate(self.stages):
            skip = features[-(stage_idx + 2)]
            x = stage(x, skip)

        x = self.dropout(x)
        x = self.cls_seg(x)

        if self.output_scale != 1:
            x = F.interpolate(
                x,
                scale_factor=self.output_scale,
                mode='bilinear',
                align_corners=self.align_corners,
            )
        return x
