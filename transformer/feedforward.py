"""Position-wise feed-forward networks."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Two-layer MLP applied independently at each position."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(self.activation(self.up(x))))


class SwiGLU(nn.Module):
    """Gated variant used by most modern LMs (Llama, PaLM).

    `d_ff` defaults to 8/3 * d_model so the parameter count matches a 4x `FeedForward`.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))
