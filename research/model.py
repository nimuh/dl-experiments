"""A DNA language model whose token mixer is picked by name, for benchmarking."""

import torch
import torch.nn as nn

from transformer import MultiHeadSelfAttention, RotaryEmbedding, TransformerLayer

from .attention import (
    ChunkedAttention,
    LinearAttention,
    NaiveAttention,
    SlidingWindowAttention,
    WindowMaskAttention,
)
from .hyena import HyenaOperator


class NoMixer(nn.Module):
    """No token mixing at all: times the rest of the model, a floor for every mixer."""

    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)


# name -> factory(d_model, n_heads, rope, window, chunk) -> token-mixing module
MIXERS = {
    "sdpa": lambda d, h, rope, w, c: MultiHeadSelfAttention(d, h, rope=rope),
    "naive": lambda d, h, rope, w, c: NaiveAttention(d, h, rope=rope),
    "chunked": lambda d, h, rope, w, c: ChunkedAttention(d, h, rope=rope, chunk=c),
    "window-mask": lambda d, h, rope, w, c: WindowMaskAttention(d, h, rope=rope, window=w),
    "sliding-window": lambda d, h, rope, w, c: SlidingWindowAttention(d, h, rope=rope, window=w),
    "linear": lambda d, h, rope, w, c: LinearAttention(d, h),
    "hyena": lambda d, h, rope, w, c: HyenaOperator(d),
    "none": lambda d, h, rope, w, c: NoMixer(),
}


class LongDNAModel(nn.Module):
    """Embedding -> n_layers x (mixer + MLP) -> logits, as in `dna.train.CausalDNALM`."""

    def __init__(
        self,
        mixer: str,
        vocab_size: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        window: int = 512,
        chunk: int = 4096,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        rope = RotaryEmbedding(d_model // n_heads)  # one shared cos/sin cache
        self.layers = nn.ModuleList(
            TransformerLayer(d_model, n_heads, attention=MIXERS[mixer](d_model, n_heads, rope, window, chunk))
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x, is_causal=is_causal)
        return self.head(self.norm(x))
