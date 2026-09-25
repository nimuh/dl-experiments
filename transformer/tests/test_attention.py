"""MultiHeadSelfAttention checked against an explicit-softmax reference."""

import torch

from reference import naive_attention
from transformer import MultiHeadSelfAttention, RotaryEmbedding

B, T, D = 2, 6, 16


def make_attn(n_heads=4, bias=False, rope=False, seed=0):
    torch.manual_seed(seed)
    rope_mod = RotaryEmbedding(D // n_heads) if rope else None
    return MultiHeadSelfAttention(D, n_heads, bias=bias, rope=rope_mod).double().eval()


def key_mask(batch=B, seq=T, keep_upto=None):
    """Bool mask (B, 1, 1, T): True on keys that may be attended to."""
    keep = torch.ones(batch, 1, 1, seq, dtype=torch.bool)
    keep[:, :, :, keep_upto or seq // 2 :] = False
    return keep


def test_matches_naive_reference():
    x = torch.randn(B, T, D, dtype=torch.double)
    for n_heads in (1, 2, 4, 8):  # d_head stays even, so RoPE applies to all of them
        for bias in (False, True):
            for rope in (False, True):
                attn = make_attn(n_heads, bias, rope)
                for mask, causal in ((None, False), (None, True), (key_mask(), False)):
                    got = attn(x, mask, causal)
                    want = naive_attention(attn, x, mask, causal)
                    assert torch.allclose(got, want, atol=1e-10), (n_heads, bias, rope, causal)


def test_causal_flag_equals_explicit_triangular_mask():
    attn = make_attn()
    x = torch.randn(B, T, D, dtype=torch.double)
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.allclose(attn(x, is_causal=True), attn(x, attn_mask=tril), atol=1e-12)


def test_gradient_of_position_t_ignores_later_positions():
    """The strong form of causality: no path at all from x[>t] to y[t]."""
    attn = make_attn()
    x = torch.randn(1, T, D, dtype=torch.double, requires_grad=True)
    t = T // 2
    attn(x, is_causal=True)[0, t].sum().backward()
    assert torch.count_nonzero(x.grad[0, t + 1 :]) == 0
    assert torch.count_nonzero(x.grad[0, : t + 1]) > 0


def test_examples_in_a_batch_do_not_interact():
    attn = make_attn()
    x = torch.randn(B, T, D, dtype=torch.double)
    batched = attn(x)
    for b in range(B):
        assert torch.allclose(batched[b], attn(x[b : b + 1])[0], atol=1e-12)


def test_masked_keys_have_no_influence():
    attn = make_attn()
    keep = key_mask(seq=T)
    x = torch.randn(B, T, D, dtype=torch.double)
    scrambled = x.clone()
    scrambled[:, T // 2 :] = torch.randn(B, T - T // 2, D, dtype=torch.double) * 100

    # Queries at masked positions still attend to visible keys, so every row is finite.
    out, out_scrambled = attn(x, keep), attn(scrambled, keep)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, : T // 2], out_scrambled[:, : T // 2], atol=1e-10)


def test_output_is_a_convex_combination_of_values():
    """With identity projections, attention can only average the inputs."""
    attn = MultiHeadSelfAttention(D, n_heads=1).double().eval()
    with torch.no_grad():
        attn.qkv.weight.zero_()
        attn.qkv.weight[2 * D :] = torch.eye(D, dtype=torch.double)  # q = k = 0, v = x
        attn.out.weight.copy_(torch.eye(D, dtype=torch.double))
    x = torch.randn(1, T, D, dtype=torch.double)
    # Uniform attention over all positions, because every logit is zero.
    assert torch.allclose(attn(x), x.mean(dim=1, keepdim=True).expand(1, T, D), atol=1e-12)


def test_gradcheck():
    attn = make_attn(n_heads=2)
    x = torch.randn(1, 4, D, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: attn(t, is_causal=True), (x,))


def test_dropout_is_active_only_in_train_mode():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(D, 4, dropout=0.5)
    x = torch.randn(B, T, D)

    attn.eval()
    assert torch.equal(attn(x), attn(x))

    attn.train()
    assert not torch.equal(attn(x), attn(x))


def test_head_count_must_divide_d_model():
    for n_heads in (3, 5, D + 1):
        try:
            MultiHeadSelfAttention(D, n_heads)
        except ValueError:
            continue
        raise AssertionError(f"n_heads={n_heads} should have been rejected")
