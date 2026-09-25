"""DNA-specific modelling pieces.

    from dna import build_dna_tokenizer

    tok = build_dna_tokenizer()                 # single bases
    tok("acgtACGTNNNN")["input_ids"]            # normalized, bracketed with [CLS]/[SEP]

    kmer = build_dna_tokenizer(k=3)             # non-overlapping 3-mers
"""

from .tokenizer import (
    BASES,
    SPECIAL_TOKENS,
    UNRESOLVED_BASE,
    build_dna_tokenizer,
    dna_normalizer,
    kmer_vocabulary,
)

__all__ = [
    "BASES",
    "UNRESOLVED_BASE",
    "SPECIAL_TOKENS",
    "build_dna_tokenizer",
    "dna_normalizer",
    "kmer_vocabulary",
]
