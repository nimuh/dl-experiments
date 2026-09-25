"""Transformer layer: attention + feed-forward, wired with norms and residuals."""

from collections.abc import Callable

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward

NormFactory = Callable[[int], nn.Module]


class TransformerLayer(nn.Module):
    """One pre-norm (default) or post-norm transformer block.

    Pre-norm keeps a clean residual path and trains deep stacks without warmup;
    post-norm is the original Vaswani et al. wiring, kept for comparison.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        pre_norm: bool = True,
        norm_layer: NormFactory = nn.LayerNorm,
        rope: nn.Module | None = None,
        attention: nn.Module | None = None,
        feedforward: nn.Module | None = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = attention or MultiHeadSelfAttention(d_model, n_heads, dropout, bias, rope)
        self.ff = feedforward or FeedForward(d_model, d_ff, dropout, bias)
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.drop = nn.Dropout(dropout)

    def _residual(self, x: torch.Tensor, sublayer: Callable, norm: nn.Module) -> torch.Tensor:
        if self.pre_norm:
            return x + self.drop(sublayer(norm(x)))
        return norm(x + self.drop(sublayer(x)))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        def attend(h: torch.Tensor) -> torch.Tensor:
            return self.attn(h, attn_mask, is_causal, pos_offset)

        x = self._residual(x, attend, self.norm1)
        return self._residual(x, self.ff, self.norm2)


class TransformerStack(nn.Module):
    """`n_layers` identical blocks, with a final norm when pre-norm is used.

    A `rope=` keyword is forwarded to every block, so one `RotaryEmbedding` (and its
    cos/sin tables) is shared by the whole stack.
    """

    def __init__(self, n_layers: int, d_model: int, n_heads: int, norm_layer: NormFactory = nn.LayerNorm, **layer_kwargs):
        super().__init__()
        pre_norm = layer_kwargs.get("pre_norm", True)
        self.layers = nn.ModuleList(
            TransformerLayer(d_model, n_heads, norm_layer=norm_layer, **layer_kwargs)
            for _ in range(n_layers)
        )
        self.norm = norm_layer(d_model) if pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask, is_causal, pos_offset)
        return self.norm(x)
