"""Layer and stack wiring: norm placement, residuals, composition, injection."""

import torch
import torch.nn as nn

from reference import naive_layer
from transformer import RotaryEmbedding, TransformerLayer, TransformerStack

B, T, D, H = 2, 6, 16, 4


def make_layer(pre_norm=True, rope=False, seed=0, **kwargs):
    torch.manual_seed(seed)
    rope_mod = RotaryEmbedding(D // H) if rope else None
    return TransformerLayer(D, H, pre_norm=pre_norm, rope=rope_mod, **kwargs).double().eval()


def test_wiring_matches_the_reference_in_both_norm_placements():
    x = torch.randn(B, T, D, dtype=torch.double)
    for pre_norm in (True, False):
        layer = make_layer(pre_norm)
        for causal in (False, True):
            assert torch.allclose(
                layer(x, is_causal=causal), naive_layer(layer, x, is_causal=causal), atol=1e-10
            ), pre_norm


def test_residual_path_is_the_identity_when_sublayers_output_zero():
    x = torch.randn(B, T, D, dtype=torch.double)

    pre = make_layer(pre_norm=True)
    with torch.no_grad():
        pre.attn.out.weight.zero_()
        pre.ff.down.weight.zero_()
    assert torch.allclose(pre(x), x, atol=1e-12)

    post = make_layer(pre_norm=False)
    with torch.no_grad():
        post.attn.out.weight.zero_()
        post.ff.down.weight.zero_()
    assert torch.allclose(post(x), post.norm2(post.norm1(x)), atol=1e-12)


def test_layer_normalises_what_it_should():
    """Pre-norm normalises the sublayer input; post-norm normalises the block output."""
    x = torch.randn(B, T, D, dtype=torch.double) * 50 + 10
    post = make_layer(pre_norm=False)
    y = post(x)
    assert torch.allclose(y.mean(-1), torch.zeros(B, T, dtype=torch.double), atol=1e-6)
    assert torch.allclose(y.std(-1, unbiased=False), torch.ones(B, T, dtype=torch.double), atol=1e-3)


def test_stack_is_the_composition_of_its_layers():
    torch.manual_seed(0)
    stack = TransformerStack(3, D, H).double().eval()
    x = torch.randn(B, T, D, dtype=torch.double)

    manual = x
    for layer in stack.layers:
        manual = layer(manual, is_causal=True)
    assert torch.allclose(stack(x, is_causal=True), stack.norm(manual), atol=1e-12)
    assert isinstance(stack.norm, nn.LayerNorm)
    assert isinstance(TransformerStack(2, D, H, pre_norm=False).norm, nn.Identity)


def test_stack_shares_one_rotary_table_across_layers():
    rope = RotaryEmbedding(D // H)
    stack = TransformerStack(3, D, H, rope=rope)
    assert all(layer.attn.rope is rope for layer in stack.layers)
    assert sum(1 for _ in stack.buffers()) == 2  # cos and sin, counted once for the stack


def test_injected_submodules_replace_the_defaults():
    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, *args, **kwargs):
            self.calls += 1
            return torch.zeros_like(x)

    attn, ff = Spy(), Spy()
    layer = TransformerLayer(D, H, attention=attn, feedforward=ff).eval()
    x = torch.randn(B, T, D)
    assert torch.allclose(layer(x), x, atol=1e-6)
    assert (attn.calls, ff.calls) == (1, 1)


def test_gradient_of_position_t_ignores_later_positions():
    layer = make_layer()
    x = torch.randn(1, T, D, dtype=torch.double, requires_grad=True)
    t = T // 2
    layer(x, is_causal=True)[0, t].sum().backward()
    assert torch.count_nonzero(x.grad[0, t + 1 :]) == 0


def test_causality_survives_a_deep_stack_with_rope():
    torch.manual_seed(0)
    stack = TransformerStack(4, D, H, rope=RotaryEmbedding(D // H)).double().eval()
    x = torch.randn(1, T, D, dtype=torch.double, requires_grad=True)
    t = 1
    stack(x, is_causal=True)[0, t].sum().backward()
    assert torch.count_nonzero(x.grad[0, t + 1 :]) == 0


def test_gradcheck():
    layer = TransformerLayer(8, 2, d_ff=8).double().eval()
    x = torch.randn(1, 4, 8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: layer(t, is_causal=True), (x,))


def test_dropout_is_active_only_in_train_mode():
    torch.manual_seed(0)
    layer = TransformerLayer(D, H, dropout=0.5)
    x = torch.randn(B, T, D)
    assert torch.equal(layer.eval()(x), layer(x))
    assert not torch.equal(layer.train()(x), layer(x))


def test_alternative_norms_and_feedforwards_run_end_to_end():
    torch.manual_seed(0)
    stack = TransformerStack(2, D, H, norm_layer=nn.RMSNorm, pre_norm=True).eval()
    x = torch.randn(B, T, D)
    assert stack(x, is_causal=True).shape == (B, T, D)
