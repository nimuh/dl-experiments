"""Feed-forward blocks: exact formulas, default widths, position independence."""

import torch

from reference import naive_feedforward, naive_swiglu
from transformer import FeedForward, SwiGLU

B, T, D = 2, 5, 16


def test_feedforward_matches_the_written_out_formula():
    torch.manual_seed(0)
    ff = FeedForward(D, bias=True).double().eval()
    x = torch.randn(B, T, D, dtype=torch.double)
    assert torch.allclose(ff(x), naive_feedforward(ff, x), atol=1e-12)


def test_swiglu_matches_the_written_out_formula():
    torch.manual_seed(0)
    ff = SwiGLU(D, bias=True).double().eval()
    x = torch.randn(B, T, D, dtype=torch.double)
    assert torch.allclose(ff(x), naive_swiglu(ff, x), atol=1e-12)


def test_default_hidden_widths():
    assert FeedForward(D).up.out_features == 4 * D
    assert SwiGLU(D).gate.out_features == int(8 * D / 3)
    assert FeedForward(D, d_ff=7).up.out_features == 7
    assert SwiGLU(D, d_ff=7).up.out_features == 7


def test_swiglu_is_parameter_matched_to_a_4x_mlp():
    mlp = sum(p.numel() for p in FeedForward(64).parameters())
    gated = sum(p.numel() for p in SwiGLU(64).parameters())
    assert abs(gated - mlp) / mlp < 0.02


def test_each_position_is_transformed_independently():
    for ff in (FeedForward(D).double().eval(), SwiGLU(D).double().eval()):
        x = torch.randn(B, T, D, dtype=torch.double)
        perturbed = x.clone()
        perturbed[:, 0] = torch.randn(B, D, dtype=torch.double)
        assert torch.allclose(ff(x)[:, 1:], ff(perturbed)[:, 1:], atol=1e-12)


def test_gradcheck():
    for ff in (FeedForward(D, d_ff=8).double(), SwiGLU(D, d_ff=8).double()):
        x = torch.randn(1, 3, D, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(ff.eval(), (x,))


def test_dropout_is_active_only_in_train_mode():
    for ff in (FeedForward(D, dropout=0.5), SwiGLU(D, dropout=0.5)):
        x = torch.randn(B, T, D)
        assert torch.equal(ff.eval()(x), ff(x))
        assert not torch.equal(ff.train()(x), ff(x))
