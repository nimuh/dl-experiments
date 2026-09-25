"""Train a small causal transformer on synthetic Markov-chain DNA.

Each run trains one model configuration and appends a JSON line to `--results`
holding the full config, the training-loss curve and periodic eval losses.
Losses are nats per base, so runs with different `--kmer` are comparable.

    cd dl-experiments
    python3 -m dna.train --steps 500 --n-layers 2 --d-model 64 --pos rope \\
        --results results/dna_train.jsonl
"""

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dna.synthetic import MarkovSource, to_strings
from dna.tokenizer import build_dna_tokenizer
from transformer import RotaryEmbedding, SwiGLU, TransformerLayer

NORMS = {"layer": nn.LayerNorm, "rms": nn.RMSNorm}


class CausalDNALM(nn.Module):
    """Token embedding -> transformer layers -> next-token logits."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        pos: str = "rope",
        norm: str = "layer",
        ffn: str = "gelu",
        pre_norm: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model) if pos == "learned" else None
        rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_len) if pos == "rope" else None
        norm_layer = NORMS[norm]
        # Built layer by layer rather than with TransformerStack: its **layer_kwargs
        # would hand one SwiGLU instance to every layer.
        self.layers = nn.ModuleList(
            TransformerLayer(
                d_model, n_heads, d_ff, dropout,
                pre_norm=pre_norm,
                norm_layer=norm_layer,
                rope=rope,
                feedforward=SwiGLU(d_model, d_ff, dropout) if ffn == "swiglu" else None,
            )
            for _ in range(n_layers)
        )
        self.norm = norm_layer(d_model) if pre_norm else nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        if self.pos_embed is not None:
            x = x + self.pos_embed(torch.arange(ids.shape[1], device=ids.device))
        for layer in self.layers:
            x = layer(x, is_causal=True)
        return self.head(self.norm(x))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m dna.train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, default=Path("results/dna_train.jsonl"),
                   help="JSONL file each run's record is appended to")
    p.add_argument("--name", default=None, help="optional label stored with the run")

    m = p.add_argument_group("model")
    m.add_argument("--d-model", type=int, default=64)
    m.add_argument("--n-heads", type=int, default=4)
    m.add_argument("--n-layers", type=int, default=2)
    m.add_argument("--d-ff", type=int, default=None, help="default: 4*d_model (gelu), 8/3*d_model (swiglu)")
    m.add_argument("--dropout", type=float, default=0.0)
    m.add_argument("--pos", choices=["rope", "learned", "none"], default="rope")
    m.add_argument("--norm", choices=list(NORMS), default="layer")
    m.add_argument("--ffn", choices=["gelu", "swiglu"], default="gelu")
    m.add_argument("--post-norm", action="store_true", help="post-norm instead of pre-norm")

    d = p.add_argument_group("data")
    d.add_argument("--markov-order", type=int, default=3, help="bases of context the chain conditions on")
    d.add_argument("--concentration", type=float, default=0.3, help="Dirichlet concentration of chain rows")
    d.add_argument("--chain-seed", type=int, default=0, help="seed of the transition table")
    d.add_argument("--seq-len", type=int, default=128, help="sequence length in bases")
    d.add_argument("--kmer", type=int, default=1, help="tokenize into non-overlapping k-mers")
    d.add_argument("--eval-sequences", type=int, default=256)

    t = p.add_argument_group("training")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--warmup", type=int, default=50, help="linear warmup steps, then cosine decay to 10%%")
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0, help="seed for model init and training data")
    t.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")

    args = p.parse_args(argv)
    if args.seq_len % args.kmer:
        p.error(f"--seq-len {args.seq_len} must be a multiple of --kmer {args.kmer}")
    if args.d_model % args.n_heads:
        p.error(f"--d-model {args.d_model} must be divisible by --n-heads {args.n_heads}")
    if args.steps < 1:
        p.error("--steps must be at least 1")
    return args


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_state() -> dict:
    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
        return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def lr_lambda(warmup: int, steps: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def train(args: argparse.Namespace) -> dict:
    """Train one model; return the run record (config and curves)."""
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    source = MarkovSource(args.markov_order, args.concentration, args.chain_seed)
    tok = build_dna_tokenizer(k=args.kmer, add_cls_sep=False)
    n_tokens = args.seq_len // args.kmer

    def encode(x: np.ndarray) -> torch.Tensor:
        ids = tok(to_strings(x), return_tensors="pt")["input_ids"]
        assert ids.shape == (len(x), n_tokens), ids.shape
        return ids.to(device)

    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng([args.chain_seed, 1])  # fixed across model seeds
    eval_x = source.sample(args.eval_sequences, args.seq_len, eval_rng)
    eval_ids = encode(eval_x)

    model = CausalDNALM(
        len(tok), n_tokens, args.d_model, args.n_heads, args.n_layers, args.d_ff,
        args.dropout, args.pos, args.norm, args.ffn, pre_norm=not args.post_norm,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.warmup, args.steps))

    def loss_per_base(ids: torch.Tensor) -> torch.Tensor:
        logits = model(ids[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()) / args.kmer

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        total = sum(
            loss_per_base(eval_ids[i : i + args.batch_size]).item() * len(eval_ids[i : i + args.batch_size])
            for i in range(0, len(eval_ids), args.batch_size)
        )
        model.train()
        return total / len(eval_ids)

    train_loss, evals = [], []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        ids = encode(source.sample(args.batch_size, args.seq_len, train_rng))
        loss = loss_per_base(ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        train_loss.append(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            evals.append({"step": step, "loss": evaluate()})
            print(
                f"step {step:>6}  train {train_loss[-1]:.4f}  eval {evals[-1]['loss']:.4f}",
                file=sys.stderr,
            )

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": args.name,
        "config": config,
        "git": git_state(),
        "device": str(device),
        "torch_version": torch.__version__,
        "n_params": sum(p.numel() for p in model.parameters()),
        "loss_unit": "nats/base",
        "uniform_loss": math.log(4),
        "final_eval_loss": evals[-1]["loss"],
        "eval_loss": evals,
        "train_loss": train_loss,
        "wall_time_s": round(time.perf_counter() - start, 3),
    }


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    record = train(args)
    append_record(args.results, record)
    print(
        f"final eval {record['final_eval_loss']:.4f} nats/base (uniform {record['uniform_loss']:.4f})"
        f" -> appended to {args.results}",
        file=sys.stderr,
    )
    return record


if __name__ == "__main__":
    main()
