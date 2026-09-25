"""A DNA tokenizer built with the HuggingFace `tokenizers` API.

The alphabet is the four bases A, C, G, T, plus `N` for an unresolved base. Tokens
are single bases by default (`k=1`) or non-overlapping k-mers (`k>1`, as in DNABERT).

The normalizer handles what real sequence files contain: soft-masked lowercase from
RepeatMasker, `U` from RNA, and the line breaks of wrapped FASTA. Anything left that
is not in the alphabet — IUPAC ambiguity codes, protein letters, digits — becomes
`[UNK]` rather than being silently dropped.
"""

import itertools

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

BASES = "ACGT"
UNRESOLVED_BASE = "N"
PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIAL_TOKENS = (PAD, UNK, CLS, SEP, MASK)


def kmer_vocabulary(
    k: int = 1, include_n: bool = True, special_tokens: tuple[str, ...] = SPECIAL_TOKENS
) -> dict[str, int]:
    """Special tokens first, then every k-mer over the alphabet in lexicographic order.

    Size is `len(special_tokens) + alphabet**k`, e.g. 5 + 4 for single bases without `N`.
    """
    if k < 1:
        raise ValueError(f"k={k} must be at least 1")
    alphabet = BASES + (UNRESOLVED_BASE if include_n else "")
    vocab = {token: i for i, token in enumerate(special_tokens)}
    for mer in itertools.product(alphabet, repeat=k):
        vocab.setdefault("".join(mer), len(vocab))
    return vocab


def dna_normalizer() -> normalizers.Normalizer:
    """Uppercase soft-masked bases, fold RNA `U` onto `T`, and strip FASTA whitespace."""
    return normalizers.Sequence(
        [normalizers.Replace(Regex(r"\s+"), "")]
        + [normalizers.Replace(lower, upper) for lower, upper in zip("acgtn", "ACGTN")]
        + [normalizers.Replace(u, "T") for u in ("u", "U")]
    )


def build_dna_tokenizer(
    k: int = 1,
    include_n: bool = True,
    model_max_length: int = 1024,
    add_cls_sep: bool = True,
) -> PreTrainedTokenizerFast:
    """Build a `PreTrainedTokenizerFast` over DNA k-mers.

    With `k > 1` the sequence is cut into non-overlapping k-mers; a trailing run
    shorter than `k` has no vocabulary entry and comes back as `[UNK]`, so pad or trim
    sequences to a multiple of `k` if that matters.

    Set `add_cls_sep=False` for causal language modelling, where the bracketing
    `[CLS]`/`[SEP]` pair is usually unwanted.
    """
    vocab = kmer_vocabulary(k, include_n)
    alphabet = BASES + (UNRESOLVED_BASE if include_n else "")

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=UNK))
    tokenizer.normalizer = dna_normalizer()
    # `isolated` keeps every match as its own token and leaves non-matching runs
    # (anything outside the alphabet) as separate pieces, which the model maps to [UNK].
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(f"[{alphabet}]{{{k}}}"), behavior="isolated"
    )
    if add_cls_sep:
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{CLS} $A {SEP}",
            pair=f"{CLS} $A {SEP} $B:1 {SEP}:1",
            special_tokens=[(CLS, vocab[CLS]), (SEP, vocab[SEP])],
        )
    tokenizer.decoder = decoders.Fuse()  # tokens concatenate; no word separators in DNA

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=model_max_length,
        pad_token=PAD,
        unk_token=UNK,
        cls_token=CLS,
        sep_token=SEP,
        mask_token=MASK,
    )
