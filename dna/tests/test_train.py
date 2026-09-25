"""The training CLI: records are appended, and a short run actually learns."""

import json
import math
import tempfile
from pathlib import Path

from dna.train import main

TINY = ["--d-model", "32", "--n-heads", "4", "--n-layers", "1", "--seq-len", "32",
        "--batch-size", "8", "--eval-sequences", "16", "--device", "cpu"]


def test_each_run_appends_one_record():
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp) / "sub" / "runs.jsonl"
        main(TINY + ["--steps", "3", "--eval-every", "2", "--results", str(results)])
        main(TINY + ["--steps", "2", "--ffn", "swiglu", "--norm", "rms", "--pos", "learned",
                     "--results", str(results)])
        records = [json.loads(line) for line in results.read_text().splitlines()]

    assert len(records) == 2
    first, second = records
    assert len(first["train_loss"]) == 3
    assert [e["step"] for e in first["eval_loss"]] == [2, 3]
    assert first["config"]["steps"] == 3 and first["config"]["ffn"] == "gelu"
    assert second["config"]["ffn"] == "swiglu" and second["config"]["pos"] == "learned"
    assert all(math.isfinite(x) for r in records for x in r["train_loss"])


def test_kmer_losses_are_per_base():
    with tempfile.TemporaryDirectory() as tmp:
        record = main(TINY + ["--steps", "1", "--kmer", "2", "--post-norm", "--pos", "none",
                              "--results", str(Path(tmp) / "r.jsonl")])
    # An untrained model is near uniform over the 30-token vocabulary: log(30)/2 per base.
    assert abs(record["train_loss"][0] - math.log(30) / 2) < 0.3


def test_short_run_learns_the_chain():
    with tempfile.TemporaryDirectory() as tmp:
        record = main(TINY + ["--steps", "150", "--markov-order", "1", "--concentration", "0.2",
                              "--eval-every", "150", "--results", str(Path(tmp) / "r.jsonl")])
    assert record["final_eval_loss"] < record["uniform_loss"]
