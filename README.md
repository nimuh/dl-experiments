# dl-experiments
## transformer/

A small, modular transformer implementation in PyTorch.

- `attention.py` — `MultiHeadSelfAttention` (fused QKV, `F.scaled_dot_product_attention`)
- `feedforward.py` — `FeedForward` (GELU MLP) and `SwiGLU`
- `layer.py` — `TransformerLayer` (pre-/post-norm) and `TransformerStack`
- `positional.py` — `RotaryEmbedding` (RoPE), applied to q/k inside attention

```python
from transformer import TransformerLayer, TransformerStack

layer = TransformerLayer(d_model=256, n_heads=8, dropout=0.1)
y = layer(x, is_causal=True)                    # x: (batch, seq, d_model)

model = TransformerStack(n_layers=6, d_model=256, n_heads=8)
```

RoPE is opt-in and one instance is shared by the whole stack (`pos_offset=` shifts the
absolute positions, for incremental decoding later):

```python
from transformer import RotaryEmbedding, TransformerStack

model = TransformerStack(6, 256, 8, rope=RotaryEmbedding(d_head=32))
y = model(x, is_causal=True)
```

Submodules are injectable, so variants are a keyword away:

```python
import torch.nn as nn
from transformer import SwiGLU, TransformerLayer

layer = TransformerLayer(256, 8, norm_layer=nn.RMSNorm, feedforward=SwiGLU(256))
```

### Tests

`tests/` checks the fast path against naive reference implementations in
`tests/reference.py` (explicit softmax, loops over batch and head, rotations written
out pair by pair), plus mask semantics, gradient-level causality, norm placement,
permutation equivariance and double-precision `gradcheck`.

```bash
cd dl-experiments/transformer && python3 tests/run_tests.py   # or: pytest tests
```

## dna/

DNA-specific pieces, starting with the tokenizer (HuggingFace `tokenizers`, wrapped as
a `PreTrainedTokenizerFast`).

- `tokenizer.py` — `build_dna_tokenizer`, `kmer_vocabulary`, `dna_normalizer`

```python
from dna import build_dna_tokenizer

tok = build_dna_tokenizer()                    # single bases: A C G T (+ N)
tok("acgt\nACGU")["input_ids"]                 # -> [CLS] A C G T A C G T [SEP]

kmer = build_dna_tokenizer(k=3)                # non-overlapping 3-mers, 125 + 5 tokens
causal = build_dna_tokenizer(add_cls_sep=False)  # no bracketing, for causal LMs
```

The vocabulary is specials (`[PAD] [UNK] [CLS] [SEP] [MASK]`, ids 0-4) followed by every
k-mer in lexicographic order. Normalization covers what sequence files actually contain:
soft-masked lowercase is uppercased, RNA `U` folds onto `T`, wrapped-FASTA line breaks are
dropped, and anything else — IUPAC ambiguity codes, digits — becomes `[UNK]` rather than
vanishing. `save_pretrained` / `from_pretrained` round-trip as usual.

### Training on synthetic DNA

`synthetic.py` defines `MarkovSource`, a random order-`m` Markov chain over ACGT.
`train.py` trains one causal LM configuration on
fresh samples each step. It then appends one JSON line to `--results` holding the config,
per-step train loss, periodic eval loss, parameter count and git commit.
All losses are in nats per base, so runs with different `--kmer` can be compared.

```bash
cd dl-experiments
python3 -m dna.train --steps 1000 --n-layers 2 --d-model 64 --pos rope --results results/dna_train.jsonl
python3 -m dna.train --steps 1000 --ffn swiglu --norm rms --pos learned --kmer 3 --results results/dna_train.jsonl
python3 -m dna.train --help   # model, data and training flags
```

```bash
cd dl-experiments/dna && python3 tests/run_tests.py   # or: pytest tests
```
