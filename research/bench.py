"""Forward-pass time vs sequence length, one JSON line per (mixer, length).

Lengths double from --min-len to --max-len. A mixer stops at the first length whose
forward pass exceeds --budget seconds or runs out of memory (recorded as such).

    cd dl-experiments
    python3 -m research.bench --out research/results/causal_fp32.jsonl
    python3 -m research.bench --mixers sdpa hyena --dtype float16 --bidirectional
"""

import os

# Fail allocations early instead of swapping the whole machine (MPS only).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from .model import MIXERS, LongDNAModel

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@torch.inference_mode()
def time_forward(model, ids, causal: bool, repeats: int, budget: float) -> list[float]:
    """Seconds per forward pass after one warm-up call; fewer repeats if slow."""
    device = ids.device
    times = []
    for i in range(repeats + 1):
        synchronize(device)
        start = time.perf_counter()
        model(ids, is_causal=causal)
        synchronize(device)
        elapsed = time.perf_counter() - start
        if i > 0:
            times.append(elapsed)
        if i > 0 and elapsed > budget:
            break  # one slow sample is enough; skip the rest
    return times


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python3 -m research.bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mixers", nargs="+", default=list(MIXERS), choices=list(MIXERS))
    p.add_argument("--min-len", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=2**20)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--window", type=int, default=512, help="sliding-window size")
    p.add_argument("--chunk", type=int, default=4096, help="query block of the chunked mixer")
    p.add_argument("--dtype", choices=list(DTYPES), default="float32")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--bidirectional", action="store_true", help="non-causal attention")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--budget", type=float, default=10.0, help="seconds; longer lengths are skipped")
    p.add_argument("--out", type=Path, default=Path("research/results/bench.jsonl"))
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device, dtype = torch.device(args.device), DTYPES[args.dtype]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lengths = [args.min_len * 2**i for i in range(64) if args.min_len * 2**i <= args.max_len]
    config = {k: v for k, v in vars(args).items() if k not in ("mixers", "out")}
    config.update(torch=torch.__version__)

    for name in args.mixers:
        torch.manual_seed(0)
        model = LongDNAModel(name, 16, args.d_model, args.n_heads, args.n_layers, args.window, args.chunk)
        model = model.to(device=device, dtype=dtype).eval()
        for T in lengths:
            ids = torch.randint(0, 16, (args.batch, T), device=device)
            record = {"mixer": name, "seq_len": T, **config}
            try:
                times = time_forward(model, ids, not args.bidirectional, args.repeats, args.budget)
                record.update(status="ok", median_s=statistics.median(times), min_s=min(times), times_s=times)
            except (RuntimeError, torch.OutOfMemoryError) as e:
                record.update(status="error", error=str(e).splitlines()[0][:200])
            del ids
            free_memory(device)
            with args.out.open("a") as f:
                f.write(json.dumps(record) + "\n")
            shown = f"{record['median_s'] * 1e3:10.1f} ms" if record["status"] == "ok" else record["error"]
            print(f"{name:>15} T={T:>8}  {shown}", flush=True)
            if record["status"] != "ok" or record["median_s"] > args.budget:
                break
        del model
        free_memory(device)


if __name__ == "__main__":
    main()
