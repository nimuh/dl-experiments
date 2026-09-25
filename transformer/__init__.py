"""A small, readable transformer implementation.

    from transformer import TransformerLayer, TransformerStack

    layer = TransformerLayer(d_model=256, n_heads=8, dropout=0.1)
    y = layer(x, is_causal=True)           # x: (batch, seq, d_model)

    model = TransformerStack(
        n_layers=6, d_model=256, n_heads=8, rope=RotaryEmbedding(d_head=32)
    )
"""

from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward, SwiGLU
from .layer import TransformerLayer, TransformerStack
from .positional import RotaryEmbedding, apply_rotary, rotate_half

__all__ = [
    "MultiHeadSelfAttention",
    "FeedForward",
    "SwiGLU",
    "RotaryEmbedding",
    "apply_rotary",
    "rotate_half",
    "TransformerLayer",
    "TransformerStack",
]
