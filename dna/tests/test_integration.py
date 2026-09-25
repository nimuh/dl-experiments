"""The tokenizer feeding the transformer: ids -> embeddings -> stack."""

import torch
import torch.nn as nn

from dna import build_dna_tokenizer
from transformer import RotaryEmbedding, TransformerStack

D, H = 32, 4


def encode_batch(tok, sequences):
    batch = tok(sequences, padding=True, return_tensors="pt")
    keep = batch["attention_mask"].bool()[:, None, None, :]  # (B, 1, 1, T) key mask
    return batch["input_ids"], keep


def test_token_ids_run_through_the_stack():
    torch.manual_seed(0)
    tok = build_dna_tokenizer()
    embed = nn.Embedding(len(tok), D, padding_idx=tok.pad_token_id)
    stack = TransformerStack(2, D, H, rope=RotaryEmbedding(D // H)).eval()

    ids, keep = encode_batch(tok, ["ACGTACGT", "ACGN", "T"])
    y = stack(embed(ids), attn_mask=keep)

    assert y.shape == (3, ids.shape[1], D)
    assert torch.isfinite(y).all()


def test_padding_does_not_leak_into_real_positions():
    torch.manual_seed(0)
    tok = build_dna_tokenizer()
    embed = nn.Embedding(len(tok), D)
    stack = TransformerStack(2, D, H).eval()

    short, long = "ACGT", "ACGTACGTACGT"
    ids, keep = encode_batch(tok, [short, long])
    with torch.no_grad():
        padded = stack(embed(ids), attn_mask=keep)
        alone = stack(embed(tok(short, return_tensors="pt")["input_ids"]))

    real = len(tok(short)["input_ids"])
    assert torch.allclose(padded[0, :real], alone[0], atol=1e-5)


def test_embedding_table_covers_every_token_id():
    tok = build_dna_tokenizer(k=3)
    embed = nn.Embedding(len(tok), D)
    ids = tok("ACGTACGTACGT", return_tensors="pt")["input_ids"]
    assert int(ids.max()) < embed.num_embeddings
    assert embed(ids).shape == (1, ids.shape[1], D)
