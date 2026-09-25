"""Sanity checks: run with `pytest` or `python test_transformer.py`."""

import torch
import torch.nn as nn

from transformer import SwiGLU, TransformerLayer, TransformerStack

B, T, D, H = 2, 16, 32, 4


def test_shape_is_preserved():
    layer = TransformerLayer(D, H).eval()
    assert layer(torch.randn(B, T, D)).shape == (B, T, D)


def test_causal_mask_hides_the_future():
    layer = TransformerLayer(D, H).eval()
    x = torch.randn(B, T, D)
    perturbed = x.clone()
    perturbed[:, T // 2 :] = torch.randn(B, T - T // 2, D)

    with torch.no_grad():
        y, y_perturbed = layer(x, is_causal=True), layer(perturbed, is_causal=True)

    prefix = slice(None, T // 2)
    assert torch.allclose(y[:, prefix], y_perturbed[:, prefix], atol=1e-5)
    assert not torch.allclose(y[:, T // 2 :], y_perturbed[:, T // 2 :])


def test_padding_mask_ignores_padded_keys():
    layer = TransformerLayer(D, H).eval()
    x = torch.randn(B, T, D)
    keep = torch.ones(B, 1, 1, T, dtype=torch.bool)
    keep[:, :, :, T // 2 :] = False  # mask out the tail as padding

    padded = x.clone()
    padded[:, T // 2 :] = 1e3  # garbage in the padded region

    with torch.no_grad():
        y, y_padded = layer(x, attn_mask=keep), layer(padded, attn_mask=keep)

    real = slice(None, T // 2)
    assert torch.allclose(y[:, real], y_padded[:, real], atol=1e-4)


def test_gradients_reach_every_parameter():
    stack = TransformerStack(n_layers=3, d_model=D, n_heads=H, dropout=0.1)
    stack(torch.randn(B, T, D), is_causal=True).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.parameters())


def test_swap_in_alternative_submodules():
    layer = TransformerLayer(
        D, H, norm_layer=nn.RMSNorm, feedforward=SwiGLU(D, dropout=0.1), pre_norm=False
    ).eval()
    assert layer(torch.randn(B, T, D)).shape == (B, T, D)


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"pass  {name}")
