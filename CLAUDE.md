# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

Two sibling top-level packages with no packaging metadata (no `pyproject.toml` / `setup.py`): they are importable only when `dl-experiments/` is on `sys.path`. Each `tests/conftest.py` inserts `dl-experiments/` and its own `tests/` dir into `sys.path`, which is what makes `from transformer import ...` and `from reference import ...` work.

- `transformer/` — PyTorch transformer building blocks (attention, FFN/SwiGLU, layer/stack, RoPE).
- `dna/` — DNA tokenizer on HuggingFace `tokenizers`, wrapped as `PreTrainedTokenizerFast`. `dna` tests import `transformer` too (`dna/tests/test_integration.py`).

Dependencies: `torch`, `tokenizers`, `transformers`. `pytest` is optional and not installed in the default env.

## Tests

Each package has its own test dir and a pytest-free runner that discovers `test_*` functions in `test_*.py` (seeded with `torch.manual_seed(0)`):

```bash
cd transformer && python3 tests/run_tests.py     # or: pytest tests
cd dna && python3 tests/run_tests.py
```

Single test without pytest (run from inside the `tests/` dir so `conftest` is importable):

```bash
cd transformer/tests && python3 -c "import conftest, test_attention as t; t.test_gradcheck()"
# with pytest: pytest tests/test_attention.py::test_gradcheck
```

Tests are plain functions with bare `assert`s — no fixtures, no pytest-specific features — so they must keep working under `run_tests.py`.

Training CLI (must run as a module from `dl-experiments/` so both packages import):

```bash
python3 -m dna.train --steps 500 --pos rope --results results/dna_train.jsonl
```

## Architecture

**Testing strategy (transformer):** `transformer/tests/reference.py` holds deliberately slow, naive reimplementations (explicit softmax, loops over batch and head, RoPE rotated pair by pair). They must share no code with the modules under test — agreement is only evidence of correctness if the reference can't inherit the same bug. Tests also cover mask semantics, gradient-level causality, norm placement, permutation equivariance and float64 `gradcheck`.

**Call signature threaded through every level:** `TransformerStack.forward` → `TransformerLayer.forward` → `MultiHeadSelfAttention.forward` all take `(x, attn_mask, is_causal, pos_offset)` and pass them straight down. `attn_mask` is a bool mask broadcastable to `(B, H, T, T)` with **True = attend** (SDPA convention); for padding, build a `(B, 1, 1, T)` key mask from the tokenizer's `attention_mask`. `pos_offset` is the absolute position of the first token, reserved for incremental decoding.

**Dependency injection:** `TransformerLayer` accepts `attention=`, `feedforward=`, `norm_layer=` (a factory `d_model -> Module`, e.g. `nn.RMSNorm`), and `rope=`. `TransformerStack` forwards `**layer_kwargs` to every layer, so passing `rope=RotaryEmbedding(...)` shares one instance (and its cos/sin cache) across the whole stack. `pre_norm=True` (default) adds a final norm in the stack; post-norm uses `nn.Identity`.

**RoPE:** applied inside attention to q/k after the head split, before SDPA. The cos/sin tables are non-persistent buffers rebuilt lazily when a longer sequence, different device, or different dtype arrives — always computed in float64 then cast. `d_head` must be even.

**DNA tokenizer:** vocab = specials `[PAD] [UNK] [CLS] [SEP] [MASK]` (ids 0–4) then every k-mer over `ACGT`(+`N`) in lexicographic order. Normalizer strips whitespace (wrapped FASTA), uppercases soft-masked bases, folds `U`→`T`. Out-of-alphabet characters become `[UNK]` rather than being dropped, and with `k>1` a trailing partial k-mer also becomes `[UNK]`. `add_cls_sep=False` for causal LMs.

**Training (`dna/train.py`, `dna/synthetic.py`):** data comes from `MarkovSource`, a random Markov chain with a known transition table (`nll(x)` gives the exact per-base NLL). The first k-mer token is context only, and losses are divided by `k`, so they are nats/base. `CausalDNALM` builds its `TransformerLayer`s itself rather than using `TransformerStack`, because `TransformerStack` forwards the same `feedforward=` instance to every layer. Each run appends one JSONL record; the eval set is seeded only by `--chain-seed`, so it stays the same across model seeds.
