"""Synthetic DNA: sequences drawn from a random Markov chain.

Each base depends on the previous `order` bases through a transition table whose rows
are drawn from a Dirichlet. Because the table is known, the exact negative
log-likelihood of any sample is available.

Smaller `concentration` makes the rows more peaked, so the chain is more predictable.
"""

import numpy as np

from .tokenizer import BASES


class MarkovSource:
    """An order-`order` Markov chain over A, C, G, T with a random transition table.

    The first `order` bases of every sample are uniform and independent; each later
    base is drawn from the row of `probs` picked by the `order` bases before it.
    """

    def __init__(self, order: int = 3, concentration: float = 0.3, seed: int = 0):
        if order < 0:
            raise ValueError(f"order={order} must be non-negative")
        if concentration <= 0:
            raise ValueError(f"concentration={concentration} must be positive")
        self.order = order
        self.n_contexts = len(BASES) ** order
        rng = np.random.default_rng(seed)
        self.probs = rng.dirichlet(np.full(len(BASES), concentration), size=self.n_contexts)
        # Base-4 place values, most recent base least significant.
        self._place = len(BASES) ** np.arange(order - 1, -1, -1)

    def sample(self, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
        """`n` sequences of `length` bases, as base indices into `BASES`, shape (n, length)."""
        if length < self.order:
            raise ValueError(f"length={length} must be at least order={self.order}")
        x = np.empty((n, length), dtype=np.int64)
        x[:, : self.order] = rng.integers(0, len(BASES), (n, self.order))
        context = x[:, : self.order] @ self._place
        cdf = self.probs.cumsum(axis=1)
        for t in range(self.order, length):
            u = rng.random(n)
            base = np.minimum((u[:, None] > cdf[context]).sum(axis=1), len(BASES) - 1)
            x[:, t] = base
            context = (context * len(BASES) + base) % self.n_contexts
        return x

    def nll(self, x: np.ndarray) -> np.ndarray:
        """Per-base negative log-likelihood (nats) of `x` under the chain, shape (n, length)."""
        out = np.full(x.shape, np.log(len(BASES)))  # the uniform prefix
        context = x[:, : self.order] @ self._place
        for t in range(self.order, x.shape[1]):
            out[:, t] = -np.log(self.probs[context, x[:, t]])
            context = (context * len(BASES) + x[:, t]) % self.n_contexts
        return out


def to_strings(x: np.ndarray) -> list[str]:
    """Base indices (n, length) -> `n` sequence strings."""
    letters = np.array(list(BASES))
    return ["".join(row) for row in letters[x]]
