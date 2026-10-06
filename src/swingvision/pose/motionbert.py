"""MotionBERT's DSTformer (2D keypoint sequence → 3D), vendored for inference.

Adapted from https://github.com/Walter0807/MotionBERT (``lib/model/DSTformer.py``),
Copyright (c) 2023 Wentao Zhu et al., Apache License 2.0. Only what inference with the
released ``pose3d`` checkpoints needs is kept: the spatial and temporal attention blocks, the
dual-stream fusion and the regression head. Parameter names match the checkpoints.

Input: ``(B, T, 17, 3)`` H36M-ordered 2D keypoints normalized to [-1, 1] plus confidence;
output ``(B, T, 17, 3)`` 3D joints in the same normalized camera space (x right, y down,
z away from the camera). ``T`` ≤ ``maxlen`` (243).
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head attention across joints (``spatial``) or across frames (``temporal``)."""

    def __init__(self, dim: int, num_heads: int, mode: str):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.mode = mode
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, seqlen: int):
        B, N, C = x.shape
        h = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, c)
        if self.mode == "spatial":
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            c = C // h
            qt = q.reshape(-1, seqlen, h, N, c).permute(0, 2, 3, 1, 4)  # (B', H, N, T, c)
            kt = k.reshape(-1, seqlen, h, N, c).permute(0, 2, 3, 1, 4)
            vt = v.reshape(-1, seqlen, h, N, c).permute(0, 2, 3, 1, 4)
            attn = ((qt @ kt.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = (attn @ vt).permute(0, 3, 2, 1, 4).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, order: str, norm_layer):
        super().__init__()
        self.order = order  # "st": spatial then temporal; "ts": the reverse
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(dim, num_heads, "spatial")
        self.attn_t = Attention(dim, num_heads, "temporal")
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_s = MLP(dim, hidden)
        self.mlp_t = MLP(dim, hidden)

    def _spatial(self, x, seqlen):
        x = x + self.attn_s(self.norm1_s(x), seqlen)
        return x + self.mlp_s(self.norm2_s(x))

    def _temporal(self, x, seqlen):
        x = x + self.attn_t(self.norm1_t(x), seqlen)
        return x + self.mlp_t(self.norm2_t(x))

    def forward(self, x, seqlen: int):
        if self.order == "st":
            return self._temporal(self._spatial(x, seqlen), seqlen)
        return self._spatial(self._temporal(x, seqlen), seqlen)


class DSTformer(nn.Module):
    def __init__(
        self,
        dim_in: int = 3,
        dim_out: int = 3,
        dim_feat: int = 256,
        dim_rep: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: float = 4,
        num_joints: int = 17,
        maxlen: int = 243,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.blocks_st = nn.ModuleList(
            [Block(dim_feat, num_heads, mlp_ratio, "st", norm_layer) for _ in range(depth)]
        )
        self.blocks_ts = nn.ModuleList(
            [Block(dim_feat, num_heads, mlp_ratio, "ts", norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(dim_feat)
        self.pre_logits = nn.Sequential(
            OrderedDict([("fc", nn.Linear(dim_feat, dim_rep)), ("act", nn.Tanh())])
        )
        self.head = nn.Linear(dim_rep, dim_out)
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for _ in range(depth)])

    def forward(self, x):
        B, F, J, _ = x.shape
        x = self.joints_embed(x.reshape(-1, J, x.shape[-1])) + self.pos_embed
        C = x.shape[-1]
        x = x.reshape(-1, F, J, C) + self.temp_embed[:, :F]
        x = x.reshape(B * F, J, C)
        for blk_st, blk_ts, att in zip(self.blocks_st, self.blocks_ts, self.ts_attn, strict=True):
            x_st = blk_st(x, F)
            x_ts = blk_ts(x, F)
            alpha = att(torch.cat([x_st, x_ts], dim=-1)).softmax(dim=-1)
            x = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]
        x = self.pre_logits(self.norm(x).reshape(B, F, J, C))
        return self.head(x)


#: The released "lite" model fine-tuned on Human3.6M for in-the-wild video (``infer_wild.py``).
LITE_CONFIG = {"dim_feat": 256, "dim_rep": 512, "depth": 5, "num_heads": 8, "mlp_ratio": 4}


def load(path, device: str = "cpu") -> DSTformer:
    model = DSTformer(**LITE_CONFIG)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_pos", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
