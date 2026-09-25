"""Each variant against a slow, independent reference (float64, CPU)."""

import torch
import torch.nn.functional as F

from research.attention import (
    ChunkedAttention,
    LinearAttention,
    NaiveAttention,
    SlidingWindowAttention,
    WindowMaskAttention,
)
from research.hyena import HyenaOperator, fft_causal_conv
from research.model import MIXERS, LongDNAModel
from transformer import MultiHeadSelfAttention, RotaryEmbedding

D, H, T = 32, 4, 37  # T deliberately not a multiple of any window or chunk


def _pair(cls, **kwargs):
    """A dense SDPA module and a `cls` module with the same weights and RoPE."""
    rope = RotaryEmbedding(D // H)
    dense = MultiHeadSelfAttention(D, H, rope=rope).double()
    other = cls(D, H, rope=rope, **kwargs).double()
    other.load_state_dict(dense.state_dict())
    return dense, other


def _banded_reference(q, k, v, window, causal):
    """Softmax attention looping over queries, each seeing |i - j| < window."""
    out = torch.zeros_like(q)
    for i in range(q.shape[-2]):
        lo, hi = max(0, i - window + 1), (i + 1 if causal else min(q.shape[-2], i + window))
        w = (q[..., i : i + 1, :] @ k[..., lo:hi, :].transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(-1)
        out[..., i : i + 1, :] = w @ v[..., lo:hi, :]
    return out


def test_exact_variants_match_sdpa():
    x = torch.randn(2, T, D, dtype=torch.float64)
    for cls, kwargs in [(NaiveAttention, {}), (ChunkedAttention, {"chunk": 8})]:
        dense, other = _pair(cls, **kwargs)
        for causal in (True, False):
            torch.testing.assert_close(other(x, is_causal=causal), dense(x, is_causal=causal))


def test_window_variants_match_banded_reference():
    q, k, v = (torch.randn(2, H, T, 8, dtype=torch.float64) for _ in range(3))
    for window in (1, 5, 16, 64):
        for cls in (SlidingWindowAttention, WindowMaskAttention):
            mod = cls(D, H, window=window)
            for causal in (True, False):
                torch.testing.assert_close(
                    mod.attend(q, k, v, causal), _banded_reference(q, k, v, window, causal)
                )


def test_sliding_window_covering_sequence_is_dense():
    dense, local = _pair(SlidingWindowAttention, window=T)
    x = torch.randn(2, T, D, dtype=torch.float64)
    for causal in (True, False):
        torch.testing.assert_close(local(x, is_causal=causal), dense(x, is_causal=causal))


def test_linear_attention_matches_quadratic_form():
    q, k, v = (torch.randn(2, H, T, 8, dtype=torch.float64) for _ in range(3))
    fq, fk = F.elu(q) + 1, F.elu(k) + 1
    mod = LinearAttention(D, H, chunk=8)
    for causal in (True, False):
        a = fq @ fk.transpose(-2, -1)
        if causal:
            a = a.tril()
        expected = (a @ v) / (a.sum(-1, keepdim=True) + mod.eps)
        torch.testing.assert_close(mod.attend(q, k, v, causal), expected)


def test_fft_conv_matches_direct_convolution():
    u, h = torch.randn(2, 3, T, dtype=torch.float64), torch.randn(3, T, dtype=torch.float64)
    direct = torch.stack([(h[:, : t + 1].flip(-1) * u[..., : t + 1]).sum(-1) for t in range(T)], dim=-1)
    torch.testing.assert_close(fft_causal_conv(u, h, group=2), direct)  # 2 groups


def test_causal_mixers_ignore_the_future():
    x = torch.randn(1, T, D, dtype=torch.float64, requires_grad=True)
    mixers = [SlidingWindowAttention(D, H, window=5), LinearAttention(D, H, chunk=8),
              ChunkedAttention(D, H, chunk=8), HyenaOperator(D)]
    for mod in mixers:
        mod.double()(x, is_causal=True)[0, 20].sum().backward()
        future, past = x.grad[0, 21:].abs().max(), x.grad[0, :21].abs().max()
        assert future < 1e-10 * past, type(mod).__name__  # FFT round-off is not exactly 0
        x.grad = None


def test_every_mixer_builds_a_model():
    ids = torch.randint(0, 16, (2, 50))
    for name in MIXERS:
        model = LongDNAModel(name, d_model=32, n_heads=4, n_layers=2, window=8, chunk=16)
        assert model(ids).shape == (2, 50, 16), name
