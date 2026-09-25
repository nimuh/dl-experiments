"""The Markov source: determinism, sampling statistics, and the likelihood."""

import math

import numpy as np

from dna import build_dna_tokenizer
from dna.synthetic import MarkovSource, to_strings


def test_same_seeds_give_the_same_chain_and_samples():
    a, b = MarkovSource(2, seed=7), MarkovSource(2, seed=7)
    assert np.array_equal(a.probs, b.probs)
    xa = a.sample(4, 50, np.random.default_rng(1))
    xb = b.sample(4, 50, np.random.default_rng(1))
    assert np.array_equal(xa, xb)
    assert not np.array_equal(a.probs, MarkovSource(2, seed=8).probs)


def test_transition_rows_are_distributions():
    src = MarkovSource(3)
    assert src.probs.shape == (64, 4)
    assert np.allclose(src.probs.sum(axis=1), 1.0)


def test_empirical_transitions_match_the_table():
    src = MarkovSource(order=1, concentration=1.0, seed=3)
    x = src.sample(2000, 100, np.random.default_rng(0))
    counts = np.zeros((4, 4))
    np.add.at(counts, (x[:, :-1].ravel(), x[:, 1:].ravel()), 1)
    empirical = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(empirical - src.probs).max() < 0.02


def test_nll_matches_a_direct_lookup():
    src = MarkovSource(order=2, seed=5)
    x = src.sample(3, 20, np.random.default_rng(0))
    nll = src.nll(x)
    for i in range(3):
        for t in range(20):
            expected = math.log(4) if t < 2 else -math.log(src.probs[4 * x[i, t - 2] + x[i, t - 1], x[i, t]])
            assert math.isclose(nll[i, t], expected)


def test_peaked_chains_have_an_nll_below_uniform():
    x_rng = np.random.default_rng(0)
    peaked = MarkovSource(order=2, concentration=0.1)
    flat = MarkovSource(order=2, concentration=100.0)
    assert peaked.nll(peaked.sample(200, 100, x_rng))[:, 2:].mean() < 0.8 * math.log(4)
    assert flat.nll(flat.sample(200, 100, x_rng))[:, 2:].mean() > 0.98 * math.log(4)


def test_order_zero_is_iid():
    src = MarkovSource(order=0, seed=2)
    x = src.sample(5000, 20, np.random.default_rng(0))
    assert np.abs(np.bincount(x.ravel(), minlength=4) / x.size - src.probs[0]).max() < 0.01


def test_strings_round_trip_through_the_tokenizer():
    x = MarkovSource(1).sample(2, 12, np.random.default_rng(0))
    tok = build_dna_tokenizer(k=1, add_cls_sep=False)
    ids = tok(to_strings(x))["input_ids"]
    assert [[i - 5 for i in row] for row in ids] == x.tolist()  # A,C,G,T follow 5 specials
