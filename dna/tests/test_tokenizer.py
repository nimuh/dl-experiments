"""Correctness tests for the DNA tokenizer."""

import random
import tempfile

from transformers import PreTrainedTokenizerFast

from dna import BASES, SPECIAL_TOKENS, build_dna_tokenizer, kmer_vocabulary

SEQ = "ACGTACGTTTGCA"


def random_sequence(length, alphabet=BASES, seed=0):
    return "".join(random.Random(seed).choices(alphabet, k=length))


def tokens_of(tok, text, **kwargs):
    return tok.convert_ids_to_tokens(tok(text, **kwargs)["input_ids"])


# --- vocabulary ------------------------------------------------------------------


def test_vocabulary_holds_the_specials_then_every_kmer():
    vocab = kmer_vocabulary(k=1)
    assert list(vocab)[: len(SPECIAL_TOKENS)] == list(SPECIAL_TOKENS)
    assert list(vocab)[len(SPECIAL_TOKENS) :] == ["A", "C", "G", "T", "N"]
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_vocabulary_size_is_specials_plus_alphabet_to_the_k():
    for k in (1, 2, 3):
        assert len(kmer_vocabulary(k, include_n=True)) == len(SPECIAL_TOKENS) + 5**k
        assert len(kmer_vocabulary(k, include_n=False)) == len(SPECIAL_TOKENS) + 4**k
        assert build_dna_tokenizer(k).vocab_size == len(SPECIAL_TOKENS) + 5**k


def test_kmers_are_distinct_and_cover_the_alphabet():
    vocab = kmer_vocabulary(k=2, include_n=False)
    mers = [t for t in vocab if t not in SPECIAL_TOKENS]
    assert len(set(mers)) == len(mers) == 16
    assert set("".join(mers)) == set(BASES)


def test_k_must_be_positive():
    for k in (0, -1):
        try:
            kmer_vocabulary(k)
        except ValueError:
            continue
        raise AssertionError(f"k={k} should have been rejected")


# --- single-base tokenization ----------------------------------------------------


def test_each_base_becomes_exactly_one_token():
    tok = build_dna_tokenizer()
    assert tokens_of(tok, SEQ, add_special_tokens=False) == list(SEQ)
    assert len({tok.convert_tokens_to_ids(b) for b in BASES}) == 4


def test_decode_recovers_the_sequence():
    tok = build_dna_tokenizer()
    for length in (1, 7, 64, 513):
        seq = random_sequence(length, seed=length)
        ids = tok(seq)["input_ids"]
        assert tok.decode(ids, skip_special_tokens=True) == seq


def test_empty_input_yields_only_the_special_tokens():
    tok = build_dna_tokenizer()
    assert tokens_of(tok, "") == ["[CLS]", "[SEP]"]


# --- normalization of real-world sequence text -----------------------------------


def test_soft_masked_lowercase_is_uppercased():
    tok = build_dna_tokenizer()
    assert tok("acgtn")["input_ids"] == tok("ACGTN")["input_ids"]
    assert tokens_of(tok, "acGTn", add_special_tokens=False) == ["A", "C", "G", "T", "N"]


def test_rna_uracil_folds_onto_thymine():
    tok = build_dna_tokenizer()
    assert tok("ACGU")["input_ids"] == tok("ACGT")["input_ids"]
    assert tok("acgu")["input_ids"] == tok("ACGT")["input_ids"]


def test_fasta_line_breaks_and_spaces_are_dropped():
    tok = build_dna_tokenizer()
    wrapped = "ACGTACGT\nTTGCA\r\n"
    assert tok(wrapped)["input_ids"] == tok(SEQ)["input_ids"]
    assert tok("ACG TAC GT")["input_ids"] == tok("ACGTACGT")["input_ids"]


def test_unrecognised_letters_become_unk_one_per_character():
    tok = build_dna_tokenizer()
    # R and Y are IUPAC ambiguity codes; the vocabulary deliberately does not have them.
    assert tokens_of(tok, "ARCYG", add_special_tokens=False) == ["A", "[UNK]", "C", "[UNK]", "G"]
    assert tokens_of(tok, "AC1GT", add_special_tokens=False) == ["A", "C", "[UNK]", "G", "T"]


def test_n_is_a_real_token_only_when_the_alphabet_includes_it():
    with_n = build_dna_tokenizer(include_n=True)
    without_n = build_dna_tokenizer(include_n=False)
    assert tokens_of(with_n, "ANA", add_special_tokens=False) == ["A", "N", "A"]
    assert tokens_of(without_n, "ANA", add_special_tokens=False) == ["A", "[UNK]", "A"]


# --- special tokens, padding, truncation -----------------------------------------


def test_sequences_are_bracketed_only_when_asked():
    bracketed = build_dna_tokenizer(add_cls_sep=True)
    plain = build_dna_tokenizer(add_cls_sep=False)
    assert tokens_of(bracketed, SEQ) == ["[CLS]", *SEQ, "[SEP]"]
    assert tokens_of(plain, SEQ) == list(SEQ)
    assert tokens_of(bracketed, SEQ, add_special_tokens=False) == list(SEQ)


def test_padding_marks_pad_positions_in_the_attention_mask():
    tok = build_dna_tokenizer()
    batch = tok(["ACGT", "AC"], padding=True)
    lengths = [len(ids) for ids in batch["input_ids"]]
    assert lengths[0] == lengths[1]
    assert batch["attention_mask"][0] == [1] * lengths[0]
    assert batch["attention_mask"][1] == [1, 1, 1, 1, 0, 0]
    assert batch["input_ids"][1][-2:] == [tok.pad_token_id] * 2
    assert tok.pad_token_id == 0


def test_truncation_respects_model_max_length():
    tok = build_dna_tokenizer(model_max_length=16)
    ids = tok(random_sequence(1000), truncation=True)["input_ids"]
    assert len(ids) == 16
    assert (ids[0], ids[-1]) == (tok.cls_token_id, tok.sep_token_id)


def test_pair_encoding_separates_the_two_sequences():
    tok = build_dna_tokenizer()
    encoded = tok("ACGT", "TTTT", return_token_type_ids=True)
    assert tok.convert_ids_to_tokens(encoded["input_ids"]) == [
        "[CLS]", *"ACGT", "[SEP]", *"TTTT", "[SEP]"
    ]
    assert encoded["token_type_ids"] == [0] * 6 + [1] * 5  # segment ids, on request


# --- k-mers ----------------------------------------------------------------------


def test_kmers_are_non_overlapping_chunks():
    tok = build_dna_tokenizer(k=3)
    assert tokens_of(tok, "ACGTACGTA", add_special_tokens=False) == ["ACG", "TAC", "GTA"]
    assert tok.decode(tok("ACGTACGTA")["input_ids"], skip_special_tokens=True) == "ACGTACGTA"


def test_a_ragged_tail_has_no_kmer_and_becomes_unk():
    tok = build_dna_tokenizer(k=3)
    assert tokens_of(tok, "ACGTAC" + "G", add_special_tokens=False) == ["ACG", "TAC", "[UNK]"]


def test_kmer_tokenizer_normalizes_like_the_base_one():
    tok = build_dna_tokenizer(k=2)
    assert tokens_of(tok, "ac\ngu", add_special_tokens=False) == ["AC", "GT"]


# --- persistence -------------------------------------------------------------------


def test_save_and_reload_preserves_behaviour():
    tok = build_dna_tokenizer(k=2, model_max_length=32)
    seq = random_sequence(40, seed=7)
    with tempfile.TemporaryDirectory() as directory:
        tok.save_pretrained(directory)
        reloaded = PreTrainedTokenizerFast.from_pretrained(directory)

    assert reloaded.get_vocab() == tok.get_vocab()
    assert reloaded(seq)["input_ids"] == tok(seq)["input_ids"]
    assert reloaded("ac\ngu")["input_ids"] == tok("ACGT")["input_ids"]
    assert reloaded.model_max_length == 32
    assert reloaded.pad_token == tok.pad_token
