from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.time_dim = int(time_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().reshape(-1, 1)
        half_dim = self.time_dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * -scale
        )
        angles = t * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.time_dim:
            embedding = F.pad(embedding, (0, self.time_dim - embedding.shape[-1]))
        return embedding


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        keep = x.new_empty((x.shape[0], 1)).bernoulli_(keep_prob)
        return x * keep / keep_prob


class DiffusionMLP(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float, drop: float = 0.0) -> None:
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SampleAwareWindowAttention(nn.Module):
    """Window attention that never mixes points from different batch items."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int = 256,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _forward_sample(self, x: torch.Tensor) -> torch.Tensor:
        num_points, channels = x.shape
        if num_points == 0:
            return x
        window_size = self.window_size
        pad_len = (window_size - num_points % window_size) % window_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        padded_points = x.shape[0]
        num_windows = padded_points // window_size
        x = x.reshape(num_windows, window_size, channels)
        qkv = self.qkv(x).reshape(
            num_windows, window_size, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q * self.scale) @ k.transpose(-2, -1)

        if pad_len:
            valid = torch.ones(padded_points, device=x.device, dtype=torch.bool)
            valid[num_points:] = False
            valid = valid.reshape(num_windows, window_size)
            attention = attention.masked_fill(
                ~valid[:, None, None, :], torch.finfo(attention.dtype).min
            )

        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(
            num_windows, window_size, channels
        )
        output = self.proj_drop(self.proj(output)).reshape(padded_points, channels)
        return output[:num_points]

    def forward(
        self, x: torch.Tensor, offset: torch.Tensor | None = None
    ) -> torch.Tensor:
        if offset is None:
            return self._forward_sample(x)
        outputs = []
        start = 0
        for end_tensor in offset:
            end = int(end_tensor.item())
            outputs.append(self._forward_sample(x[start:end]))
            start = end
        if start != x.shape[0]:
            raise ValueError("The final offset must equal the number of point features.")
        return torch.cat(outputs, dim=0) if outputs else x


class PointwiseDiffusionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = SampleAwareWindowAttention(
            channels=channels, num_heads=num_heads, window_size=window_size
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = DiffusionMLP(channels, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.time_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, channels * 4)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_feature: torch.Tensor,
        offset: torch.Tensor | None,
    ) -> torch.Tensor:
        scale1, shift1, scale2, shift2 = self.time_modulation(time_feature).chunk(4, dim=-1)
        residual = x
        x = self.norm1(x) * (1.0 + scale1) + shift1
        x = residual + self.drop_path(self.attention(x, offset))
        residual = x
        x = self.norm2(x) * (1.0 + scale2) + shift2
        return residual + self.drop_path(self.mlp(x))


class PointwiseDiffusionDenoiser(nn.Module):
    """Noise predictor conditioned on one diffusion timestep per point."""

    def __init__(
        self,
        feat_dim: int,
        time_dim: int = 64,
        hidden_dim: int | None = None,
        num_heads: int = 4,
        num_blocks: int = 2,
        mlp_ratio: float = 4.0,
        window_size: int = 256,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or feat_dim * 4)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_projection = nn.Linear(feat_dim, hidden_dim)
        rates = [drop_path * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            [
                PointwiseDiffusionBlock(
                    channels=hidden_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=rates[i],
                )
                for i in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, feat_dim)
        self.apply(self._init_weights)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        feature: torch.Tensor,
        timestep: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if feature.dim() != 2:
            raise ValueError("feature must have shape [N, C].")
        timestep = timestep.reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(feature.shape[0])
        if timestep.numel() != feature.shape[0]:
            raise ValueError("Expected one timestep per point.")

        time_feature = self.time_mlp(self.time_embedding(timestep.to(feature.device)))
        hidden = self.input_projection(feature)
        for block in self.blocks:
            hidden = block(hidden, time_feature, offset)
        return self.output_projection(self.final_norm(hidden))
