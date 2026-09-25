"""Positional encodings."""

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) -> (-x2, x1) over the channel halves."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each channel pair of `x` (..., T, d_head) by the per-position angle."""
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (Su et al., 2021).

    Rotates query and key channel pairs by an angle proportional to their position,
    so the attention logit q_m . k_n depends on the offset (m - n) rather than on the
    absolute positions. Applied per head, after the q/k/v split and before attention.

    One instance can be shared by every layer of a stack. The cos/sin tables are
    non-persistent buffers, rebuilt from `base` whenever a longer sequence, another
    device or another dtype shows up — always computed in float64 first, so a table
    for a low-precision dtype is a rounded exact angle rather than a compounded one.
    """

    def __init__(self, d_head: int, base: float = 10_000.0, max_seq_len: int = 2048):
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head={d_head} must be even for rotary embeddings")
        self.d_head, self.base = d_head, base
        self._build_cache(max_seq_len, torch.device("cpu"), torch.float32)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.cached_len, self.cached_device, self.cached_dtype = seq_len, device, dtype
        # Built on the CPU: some backends (MPS) have no float64.
        exponent = torch.arange(0, self.d_head, 2, dtype=torch.float64) / self.d_head
        inv_freq = 1.0 / self.base**exponent
        angles = torch.outer(torch.arange(seq_len, dtype=torch.float64), inv_freq)
        angles = torch.cat((angles, angles), dim=-1)  # (T, d_head), halves share angles
        self.register_buffer("cos", angles.cos().to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(device=device, dtype=dtype), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate `q` and `k`, shaped (B, H, T, d_head), starting at absolute `offset`."""
        end = offset + q.shape[-2]
        if end > self.cached_len or (q.device, q.dtype) != (self.cached_device, self.cached_dtype):
            self._build_cache(max(end, self.cached_len), q.device, q.dtype)
        cos, sin = self.cos[offset:end], self.sin[offset:end]
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
