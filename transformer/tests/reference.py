"""Naive reference implementations.

Deliberately slow and written straight from the equations: explicit loops over
batch and head, softmax spelled out. The point is that they share no code with
the module under test, so agreement is evidence of correctness rather than of a
shared bug.
"""

import math

import torch
import torch.nn.functional as F


def linear(x: torch.Tensor, layer: torch.nn.Linear) -> torch.Tensor:
    y = x @ layer.weight.T
    return y + layer.bias if layer.bias is not None else y


def naive_attention(
    attn: torch.nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    pos_offset: int = 0,
) -> torch.Tensor:
    """Recompute `attn(x, ...)` one (batch, head) pair at a time."""
    B, T, D = x.shape
    H, dh = attn.n_heads, attn.d_head
    q, k, v = linear(x, attn.qkv).split(D, dim=-1)

    if attn.rope is not None:
        cos, sin = _rotary_tables(attn.rope, T, pos_offset, x.dtype)

    out = torch.zeros_like(x)
    for b in range(B):
        for h in range(H):
            head = slice(h * dh, (h + 1) * dh)
            qi, ki, vi = q[b, :, head], k[b, :, head], v[b, :, head]
            if attn.rope is not None:
                qi, ki = _rotate(qi, cos, sin), _rotate(ki, cos, sin)

            scores = (qi @ ki.T) / math.sqrt(dh)
            if is_causal:
                future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(future, float("-inf"))
            if attn_mask is not None:
                keep = torch.broadcast_to(attn_mask, (B, H, T, T))[b, h]
                scores = scores.masked_fill(~keep, float("-inf"))

            out[b, :, head] = torch.softmax(scores, dim=-1) @ vi

    return linear(out, attn.out)


def _rotary_tables(rope, seq_len: int, offset: int, dtype: torch.dtype):
    """Rebuild RoPE's angles straight from `base`, without touching its cache."""
    angles = torch.tensor(
        [
            [pos / rope.base ** (2 * i / rope.d_head) for i in range(rope.d_head // 2)]
            for pos in range(offset, offset + seq_len)
        ],
        dtype=torch.float64,
    )
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs (i, i + dh/2) by the position angle, one pair at a time."""
    half = x.shape[-1] // 2
    out = torch.empty_like(x)
    for i in range(half):
        a, b = x[:, i], x[:, i + half]
        out[:, i] = a * cos[:, i] - b * sin[:, i]
        out[:, i + half] = b * cos[:, i] + a * sin[:, i]
    return out


def naive_feedforward(ff: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return linear(F.gelu(linear(x, ff.up)), ff.down)


def naive_swiglu(ff: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return linear(F.silu(linear(x, ff.gate)) * linear(x, ff.up), ff.down)


def naive_layer(
    layer: torch.nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """The textbook wiring, written out in both norm placements."""
    attn = lambda h: naive_attention(layer.attn, h, attn_mask, is_causal)  # noqa: E731
    ff = lambda h: naive_feedforward(layer.ff, h)  # noqa: E731

    if layer.pre_norm:
        x = x + attn(layer.norm1(x))
        return x + ff(layer.norm2(x))
    x = layer.norm1(x + attn(x))
    return layer.norm2(x + ff(x))
