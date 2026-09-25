"""Hyena operator (Poli et al., 2023), the attention replacement used by HyenaDNA.

Attention is swapped for gated long convolutions whose filters span the whole
sequence. The convolution runs through the FFT, so the cost is O(T log T) and no
(T, T) matrix exists. This is a compact order-2 version.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fft_causal_conv(u: torch.Tensor, h: torch.Tensor, group: int = 16) -> torch.Tensor:
    """y[..., t] = sum_{s<=t} h[..., s] u[..., t-s], for u (B, D, T) and h (D, T).

    Runs `group` channels at a time: on MPS the FFT workspace grows with the whole
    batch, and small groups are both leaner and slightly faster.
    """
    T = u.shape[-1]
    n = 2 * T  # zero padding turns the FFT's circular convolution into a linear one
    return torch.cat([
        torch.fft.irfft(torch.fft.rfft(u[:, i : i + group], n=n) * torch.fft.rfft(h[i : i + group], n=n), n=n)[..., :T]
        for i in range(0, u.shape[1], group)
    ], dim=1)


class ImplicitFilter(nn.Module):
    """A (d_model, T) filter produced by an MLP over positional features, for any T.

    An exponentially decaying window (a different rate per channel) biases channels
    towards local or global mixing.
    """

    def __init__(self, d_model: int, n_bands: int = 8, hidden: int = 64):
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, n_bands + 1).float(), persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_bands + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.decay = nn.Parameter(torch.linspace(math.log(0.3), math.log(30.0), d_model))

    def forward(self, T: int, device, dtype) -> torch.Tensor:
        t = torch.linspace(0, 1, T, device=device, dtype=dtype)[:, None]  # (T, 1)
        angle = 2 * math.pi * t * self.freqs.to(dtype)
        h = self.mlp(torch.cat((t, angle.sin(), angle.cos()), dim=-1))   # (T, D)
        window = torch.exp(-self.decay.exp().to(dtype) * t)              # (T, D)
        return (h * window).transpose(0, 1)


class HyenaOperator(nn.Module):
    """Order-2 Hyena: y = x2 * conv(h, x1 * v), with short convs on the projections.

    Always causal. Takes the attention call signature so it can be passed as
    `TransformerLayer(attention=...)`; the mask/causal/offset arguments are ignored.
    """

    def __init__(self, d_model: int, short_kernel: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.short_conv = nn.Conv1d(3 * d_model, 3 * d_model, short_kernel, groups=3 * d_model,
                                    padding=short_kernel - 1)
        self.filter = ImplicitFilter(d_model)
        self.skip = nn.Parameter(torch.randn(d_model) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None, is_causal=True, pos_offset=0):
        T = x.shape[1]
        z = self.short_conv(self.in_proj(x).transpose(1, 2))[..., :T]  # causal: drop the tail
        v, x1, x2 = z.chunk(3, dim=1)                                   # each (B, D, T)
        u = x1 * v
        fft_dtype = torch.promote_types(x.dtype, torch.float32)  # no half-precision FFT
        h = self.filter(T, x.device, x.dtype).to(fft_dtype)
        y = fft_causal_conv(u.to(fft_dtype), h).to(u.dtype) + self.skip[:, None] * u
        return self.out_proj((x2 * y).transpose(1, 2))
