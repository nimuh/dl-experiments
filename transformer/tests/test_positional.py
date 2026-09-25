"""RoPE: the rotation itself, and the relative-position property it exists for."""

import math

import torch

from transformer import MultiHeadSelfAttention, RotaryEmbedding, apply_rotary, rotate_half

DH, L = 8, 12


def rope_pair(rope, q, k, offset=0):
    """Rotate (T, d_head) tensors through the module's (B, H, T, d_head) interface."""
    rq, rk = rope(q[None, None], k[None, None], offset=offset)
    return rq[0, 0], rk[0, 0]


def test_rotate_half_swaps_and_negates_the_halves():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rotation_is_the_identity_at_position_zero():
    rope = RotaryEmbedding(DH).double()
    q = torch.randn(1, DH, dtype=torch.double)
    rq, _ = rope_pair(rope, q, q)
    assert torch.allclose(rq, q, atol=1e-12)


def test_rotation_preserves_vector_norm():
    rope = RotaryEmbedding(DH).double()
    q = torch.randn(L, DH, dtype=torch.double)
    rq, _ = rope_pair(rope, q, q)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-12)


def test_logits_depend_only_on_relative_distance():
    """The defining property: <rot(q, m), rot(k, n)> is a function of m - n alone."""
    rope = RotaryEmbedding(DH).double()
    torch.manual_seed(0)
    q, k = torch.randn(DH, dtype=torch.double), torch.randn(DH, dtype=torch.double)
    tiled_q, tiled_k = q.expand(L, DH), k.expand(L, DH)
    rq, rk = rope_pair(rope, tiled_q, tiled_k)

    logits = rq @ rk.T  # (m, n)
    for distance in range(-L + 1, L):
        diag = torch.diagonal(logits, offset=-distance)
        assert torch.allclose(diag, diag[0].expand_as(diag), atol=1e-10), distance


def test_matches_the_closed_form_rotation():
    rope = RotaryEmbedding(DH, base=10_000.0).double()
    q = torch.randn(L, DH, dtype=torch.double)
    rq, _ = rope_pair(rope, q, q)

    half = DH // 2
    for pos in range(L):
        for i in range(half):
            angle = pos / (10_000.0 ** (2 * i / DH))
            a, b = q[pos, i], q[pos, i + half]
            assert math.isclose(rq[pos, i].item(), a * math.cos(angle) - b * math.sin(angle), abs_tol=1e-10)
            assert math.isclose(rq[pos, i + half].item(), b * math.cos(angle) + a * math.sin(angle), abs_tol=1e-10)


def test_offset_equals_slicing_a_longer_sequence():
    rope = RotaryEmbedding(DH).double()
    q = torch.randn(2 * L, DH, dtype=torch.double)
    full, _ = rope_pair(rope, q, q)
    shifted, _ = rope_pair(rope, q[L:], q[L:], offset=L)
    assert torch.allclose(full[L:], shifted, atol=1e-12)


def test_cache_grows_past_max_seq_len():
    rope = RotaryEmbedding(DH, max_seq_len=4).double()
    q = torch.randn(64, DH, dtype=torch.double)
    rq, _ = rope_pair(rope, q, q)
    assert rope.cached_len >= 64
    assert torch.isfinite(rq).all()

    fresh = RotaryEmbedding(DH, max_seq_len=128).double()
    assert torch.allclose(rq, rope_pair(fresh, q, q)[0], atol=1e-12)


def test_tables_follow_the_input_dtype():
    rope = RotaryEmbedding(DH)  # built in float32
    q = torch.randn(1, 1, L, DH, dtype=torch.double)
    assert rope(q, q)[0].dtype == torch.double
    assert rope.cos.dtype == torch.double


def test_apply_rotary_broadcasts_over_batch_and_heads():
    rope = RotaryEmbedding(DH).double()
    q = torch.randn(3, 2, L, DH, dtype=torch.double)
    rq, _ = rope(q, q)
    for b in range(3):
        for h in range(2):
            assert torch.allclose(rq[b, h], rope_pair(rope, q[b, h], q[b, h])[0], atol=1e-12)


def test_rope_breaks_permutation_equivariance():
    """Without positions attention is order-blind; with RoPE it is not."""
    torch.manual_seed(0)
    x = torch.randn(1, L, DH * 2, dtype=torch.double)
    perm = torch.randperm(L)

    plain = MultiHeadSelfAttention(DH * 2, 2).double().eval()
    assert torch.allclose(plain(x)[0, perm], plain(x[:, perm])[0], atol=1e-10)

    positioned = MultiHeadSelfAttention(DH * 2, 2, rope=RotaryEmbedding(DH)).double().eval()
    assert not torch.allclose(positioned(x)[0, perm], positioned(x[:, perm])[0], atol=1e-6)


def test_odd_head_dimension_is_rejected():
    try:
        RotaryEmbedding(7)
    except ValueError:
        return
    raise AssertionError("odd d_head should have been rejected")


def test_tables_build_on_a_device_without_float64():
    if not torch.backends.mps.is_available():
        return  # only meaningful where MPS exists
    rope = RotaryEmbedding(8)
    q = torch.randn(1, 2, 5, 8, device="mps")
    q_rot, _ = rope(q, q)
    cpu_rot, _ = RotaryEmbedding(8)(q.cpu(), q.cpu())
    assert torch.allclose(q_rot.cpu(), cpu_rot, atol=1e-6)
