"""Multi-head self-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention over `n_heads` parallel heads.

    Shapes: (B, T, d_model) -> (B, T, d_model). Pass a `RotaryEmbedding` as `rope`
    to make the attention logits depend on relative position.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        rope: nn.Module | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        # One fused projection for q, k, v; cheaper than three separate matmuls.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.rope = rope

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        """`attn_mask` is a bool mask broadcastable to (B, H, T, T); True = attend.

        `pos_offset` is the absolute position of the first token, for rotary embeddings.
        """
        B, T, _ = x.shape
        q, k, v = (self._split_heads(t) for t in self.qkv(x).chunk(3, dim=-1))
        if self.rope is not None:
            q, k = self.rope(q, k, offset=pos_offset)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # (B, H, T, d_head)

        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y)
