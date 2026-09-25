"""Plot forward time vs sequence length and print a markdown summary table.

    cd dl-experiments
    python3 -m research.plot research/results/causal_fp32_v2.jsonl
    python3 -m research.plot --summary research/results     # 2x2 overview figure
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order (never cycled): a mixer keeps its color in every figure.
COLORS = {
    "sdpa": "#2a78d6", "naive": "#eb6834", "chunked": "#1baf7a", "window-mask": "#eda100",
    "sliding-window": "#e87ba4", "linear": "#008300", "hyena": "#4a3aa7",
    "none": "#8a8984",
}
QUADRATIC = {"sdpa", "naive", "chunked", "window-mask"}  # drawn dashed


def load(path: Path) -> dict[str, dict[int, float]]:
    """mixer -> {seq_len: median seconds}, successful runs only (last one wins)."""
    runs = defaultdict(dict)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            runs[r["mixer"]][r["seq_len"]] = r["median_s"]
    return runs


def slope(points: dict[int, float], last: int = 3) -> float:
    """log-log slope over the longest `last` lengths: ~1 linear, ~2 quadratic."""
    xs = sorted(points)[-last:]
    if len(xs) < 2:
        return float("nan")
    lx = [math.log(x) for x in xs]
    ly = [math.log(points[x]) for x in xs]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def draw(ax, runs, title: str) -> None:
    """Forward time vs length on log-log axes, one line per mixer."""
    for name, pts in runs.items():
        xs = sorted(pts)
        ax.plot(xs, [pts[x] * 1e3 for x in xs], color=COLORS.get(name), lw=2, marker="o", ms=4,
                ls="--" if name in QUADRATIC else ":" if name == "none" else "-", label=name)
    ax.set_xscale("log", base=2)
    ax.set(yscale="log", xlabel="sequence length (tokens)", ylabel="forward pass (ms)", title=title)
    style(ax)


def style(ax) -> None:
    ax.grid(True, which="major", color="#e5e4df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(runs, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    draw(ax, runs, title)
    ax.legend(frameon=False, fontsize=8, title="dashed: O(T²) exact; dotted: no mixer", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(results: Path) -> Path:
    """One 2x2 figure: causal fp32, causal fp16, bidirectional fp32, window sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [("causal_fp32_v2", "Causal, fp32"), ("causal_fp16", "Causal, fp16"),
              ("bidir_fp32", "Bidirectional, fp32")]
    for ax, (stem, title) in zip(axes.flat, panels):
        draw(ax, load(results / f"{stem}.jsonl"), title)

    # Window sweep at T = 131k, with the 512 point and the no-mixer floor from the main run.
    ax, main = axes[1, 1], load(results / "causal_fp32_v2.jsonl")
    T = 131072
    sweep = {json.loads(l)["window"]: json.loads(l)["median_s"] for l in (results / "window_sweep.jsonl").read_text().splitlines()}
    sweep[512] = main["sliding-window"][T]
    ws = sorted(sweep)
    ax.plot(ws, [sweep[w] * 1e3 for w in ws], color=COLORS["sliding-window"], lw=2, marker="o", ms=5)
    for name in ("sdpa", "none"):
        ax.axhline(main[name][T] * 1e3, color=COLORS[name], lw=1.5, ls="--" if name == "sdpa" else ":")
        ax.annotate(name, (ws[0], main[name][T] * 1e3), xytext=(0, 4), textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_xscale("log", base=2)
    ax.set(yscale="log", xlabel="window (tokens)", ylabel="forward pass (ms)", title="Sliding-window size, causal fp32, T = 131k")
    style(ax)

    handles = {h.get_label(): h for a in axes.flat[:3] for h in a.get_lines()}
    fig.legend([handles[n] for n in COLORS if n in handles], [n for n in COLORS if n in handles],
               loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.suptitle("Forward-pass time vs sequence length (4 layers, d_model 256, batch 1, Apple M5 / MPS)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = results / "summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def table(runs) -> str:
    lengths = sorted({t for pts in runs.values() for t in pts})
    names = list(runs)
    rows = ["| T | " + " | ".join(names) + " |", "|---:|" + "---:|" * len(names)]
    for t in lengths:
        cells = [f"{runs[n][t] * 1e3:.0f}" if t in runs[n] else "—" for n in names]
        rows.append(f"| {t:,} | " + " | ".join(cells) + " |")
    rows.append("| slope | " + " | ".join(f"{slope(runs[n]):.2f}" for n in names) + " |")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="*")
    p.add_argument("--summary", type=Path, metavar="DIR", help="write DIR/summary.png from the standard runs")
    args = p.parse_args()
    if args.summary:
        print(summary(args.summary))
    for path in args.results:
        runs = load(path)
        plot(runs, path.stem, path.with_suffix(".png"))
        print(f"### {path.stem} (ms per forward; slope = log-log over last 3 lengths)\n\n{table(runs)}\n")


if __name__ == "__main__":
    main()
