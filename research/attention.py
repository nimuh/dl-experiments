"""Attention variants for long sequences, as drop-ins for `MultiHeadSelfAttention`.

Every class subclasses the repo's `MultiHeadSelfAttention` and only replaces the core
`attend(q, k, v, is_causal)` step, so projections, RoPE and weights are identical
across variants (a state dict loads into any of them).

    exact, O(T^2) time:   NaiveAttention, WindowMaskAttention, ChunkedAttention
    approximate, O(T):    SlidingWindowAttention (exact within its window), LinearAttention
"""

import torch
import torch.nn.functional as F

from transformer import MultiHeadSelfAttention


class _CoreAttention(MultiHeadSelfAttention):
    """Shared projections + RoPE; subclasses implement `attend` on (B, H, T, d_head)."""

    def forward(self, x, attn_mask=None, is_causal=False, pos_offset=0):
        if attn_mask is not None:
            raise NotImplementedError(f"{type(self).__name__} does not take attn_mask")
        B, T, _ = x.shape
        q, k, v = (self._split_heads(t) for t in self.qkv(x).chunk(3, dim=-1))
        if self.rope is not None:
            q, k = self.rope(q, k, offset=pos_offset)
        y = self.attend(q, k, v, is_causal)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))

    def attend(self, q, k, v, is_causal):
        raise NotImplementedError


class NaiveAttention(_CoreAttention):
    """Textbook attention that materialises the (T, T) score matrix: O(T^2) memory."""

    def attend(self, q, k, v, is_causal):
        scores = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
        if is_causal:
            T = q.shape[-2]
            future = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        return scores.softmax(dim=-1) @ v


class WindowMaskAttention(_CoreAttention):
    """Sliding-window attention written as a dense banded mask passed to SDPA.

    Same output as `SlidingWindowAttention`, but the kernel still visits all T^2
    pairs: masking alone does not save compute.
    """

    def __init__(self, *args, window: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window

    def attend(self, q, k, v, is_causal):
        pos = torch.arange(q.shape[-2], device=q.device)
        rel = pos[:, None] - pos[None, :]  # query - key
        mask = (rel >= 0) & (rel < self.window) if is_causal else rel.abs() < self.window
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class ChunkedAttention(_CoreAttention):
    """Exact attention computed one block of queries at a time.

    Peak memory is O(chunk * T) even for kernels that would materialise (T, T), and a
    causal query block only reads the keys up to its own end, halving the work.
    """

    def __init__(self, *args, chunk: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk = chunk

    def attend(self, q, k, v, is_causal):
        T = q.shape[-2]
        out = torch.empty_like(q)
        for s in range(0, T, self.chunk):
            e = min(s + self.chunk, T)
            end, mask = T, None
            if is_causal:
                end = e
                mask = torch.arange(s, e, device=q.device)[:, None] >= torch.arange(e, device=q.device)
            out[..., s:e, :] = F.scaled_dot_product_attention(
                q[..., s:e, :], k[..., :end, :], v[..., :end, :], attn_mask=mask
            )
        return out


class SlidingWindowAttention(_CoreAttention):
    """Each query sees keys less than `window` positions away: O(T * window).

    The sequence is cut into blocks of `window` tokens; a block's queries attend to
    their own block and its neighbour(s) (previous only if causal), and one banded
    mask shared by every block trims that to the exact window. Stacking L layers
    gives a receptive field of about L * window tokens.
    """

    def __init__(self, *args, window: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window

    def attend(self, q, k, v, is_causal):
        B, H, T, D = q.shape
        w = self.window
        n = -(-T // w)  # number of blocks
        span = 2 * w if is_causal else 3 * w
        back = n * w - T + (0 if is_causal else w)

        def windows(t):  # (B, H, T, D) -> (B*H, n, span, D): overlapping, stride one block
            t = F.pad(t, (0, 0, w, back))
            return t.unfold(-2, span, w).transpose(-2, -1).reshape(B * H, n, span, D)

        qb = F.pad(q, (0, 0, 0, n * w - T)).reshape(B * H, n, w, D)
        y = F.scaled_dot_product_attention(qb, windows(k), windows(v), attn_mask=self._band(w, span, w, 0, is_causal, q.device))
        y = y.reshape(B, H, n * w, D)[..., :T, :]

        # The shared mask lets edge blocks attend to zero padding; recompute them exactly.
        edges = {0} if is_causal else {0, max(0, n - 2), n - 1}
        rows, fixed = [], []
        for s in (b * w for b in sorted(edges)):
            e, ks = min(s + w, T), max(0, s - w + 1)
            ke = e if is_causal else min(T, e + w - 1)
            mask = self._band(e - s, ke - ks, s, ks, is_causal, q.device)
            rows.append(torch.arange(s, e, device=q.device))
            fixed.append(F.scaled_dot_product_attention(
                q[..., s:e, :], k[..., ks:ke, :], v[..., ks:ke, :], attn_mask=mask
            ))
        return y.index_copy(2, torch.cat(rows), torch.cat(fixed, dim=2))  # out of place, for autograd

    def _band(self, n_q, n_k, q_start, k_start, is_causal, device):
        """(n_q, n_k) window mask for queries at q_start.. and keys at k_start.. ."""
        rel = (torch.arange(n_q, device=device) + q_start)[:, None] - (torch.arange(n_k, device=device) + k_start)
        return (rel >= 0) & (rel < self.window) if is_causal else rel.abs() < self.window


class LinearAttention(_CoreAttention):
    """Kernelised attention (Katharopoulos et al., 2020): softmax(q k^T) -> phi(q) phi(k)^T.

    With phi = elu + 1, (phi(q) phi(k)^T) v = phi(q) (phi(k)^T v), so the (T, T) matrix
    is never formed: O(T * d_head^2). The causal case uses the chunked form: quadratic
    inside each chunk, a running (d_head, d_head) state carried across chunks. RoPE is
    not applied (it would break the positivity the normaliser relies on); causality is
    the only source of order information.
    """

    def __init__(self, *args, chunk: int = 64, eps: float = 1e-6, **kwargs):
        kwargs.pop("rope", None)
        super().__init__(*args, **kwargs)
        self.chunk, self.eps = chunk, eps

    def attend(self, q, k, v, is_causal):
        q, k = F.elu(q) + 1, F.elu(k) + 1
        if not is_causal:
            num = q @ (k.transpose(-2, -1) @ v)
            den = q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)
            return num / (den + self.eps)

        B, H, T, D = q.shape
        C = self.chunk
        n = -(-T // C)
        # Zero padding at the end is harmless: padded keys sit after every real query.
        qc, kc, vc = (F.pad(t, (0, 0, 0, n * C - T)).view(B, H, n, C, D) for t in (q, k, v))

        kv = kc.transpose(-2, -1) @ vc                  # (B, H, n, D, D) per-chunk state
        state = kv.cumsum(dim=2) - kv                    # exclusive: chunks strictly before
        ksum = kc.sum(dim=-2)                            # (B, H, n, D)
        z = ksum.cumsum(dim=2) - ksum

        scores = (qc @ kc.transpose(-2, -1)).tril()      # within-chunk, causal
        num = qc @ state + scores @ vc
        den = (qc * z.unsqueeze(-2)).sum(-1) + scores.sum(-1)
        y = num / (den.unsqueeze(-1) + self.eps)
        return y.view(B, H, n * C, D)[..., :T, :]
