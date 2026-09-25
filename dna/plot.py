"""Plot training curves from a ``dna.train`` results file, one line per model configuration.

Runs whose configs differ only in ``--seed`` (or in bookkeeping fields such as
``--results``) are grouped into one configuration: the line is the mean over
seeds and the shaded band spans min to max. Line labels are built from the
config fields that actually vary across the file.

Two panels:
  (a) eval loss;
  (b) training loss (debiased EMA), with the uniform loss marked.

    python3 -m dna.plot results/dna_train_exp.jsonl
    python3 -m dna.plot results/dna_train_exp.jsonl --out figures/depth.pdf --width 3.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Config fields that never distinguish one model configuration from another.
IGNORED_FIELDS = {"seed", "results", "name", "device", "eval_every"}

# Human-readable labels for the fields most likely to be swept.
FIELD_LABELS = {
    "n_layers": lambda v: f"{v} layer" + ("s" if v != 1 else ""),
    "d_model": lambda v: f"$d_{{\\mathrm{{model}}}}$={v}",
    "n_heads": lambda v: f"{v} head" + ("s" if v != 1 else ""),
    "d_ff": lambda v: f"$d_{{\\mathrm{{ff}}}}$={v}",
    "pos": lambda v: {"rope": "RoPE", "learned": "learned pos.", "none": "no pos."}.get(v, v),
    "norm": lambda v: f"{v}norm",
    "ffn": lambda v: {"gelu": "GELU", "swiglu": "SwiGLU"}.get(v, v),
    "post_norm": lambda v: "post-norm" if v else "pre-norm",
    "lr": lambda v: f"lr={v:g}",
    "kmer": lambda v: f"k={v}",
}

# Categorical palette, assigned in this fixed order (validated for colorblind separation
# between adjacent slots). Markers give a second, non-color encoding for print.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def config_key(run: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in run["config"].items() if k not in IGNORED_FIELDS))


def group_runs(runs: list[dict]) -> tuple[list[tuple[dict, list[dict]]], list[str]]:
    """Group runs by configuration; return groups sorted by the varying fields, and those fields."""
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        groups.setdefault(config_key(run), []).append(run)
    configs = [dict(key) for key in groups]
    varying = [k for k in configs[0] if len({repr(c.get(k)) for c in configs}) > 1]

    def sort_key(item):
        return tuple((v is None, v if v is not None else 0) for v in (dict(item[0]).get(k) for k in varying))

    ordered = sorted(groups.items(), key=sort_key)
    return [(dict(key), members) for key, members in ordered], varying


def label_for(config: dict, runs: list[dict], varying: list[str]) -> str:
    parts = [FIELD_LABELS.get(k, lambda v, k=k: f"{k}={v}")(config[k]) for k in varying]
    label = ", ".join(parts) if parts else "model"
    label += f" ({runs[0]['n_params'] / 1e3:.0f}K params)"
    if len(runs) > 1:
        label += f", {len(runs)} seeds"
    return label


def ema(x: np.ndarray, beta: float) -> np.ndarray:
    """Exponential moving average with bias correction, so early steps aren't pulled toward 0."""
    if beta <= 0:
        return x
    out, acc = np.empty_like(x), 0.0
    for i, v in enumerate(x):
        acc = beta * acc + (1 - beta) * v
        out[i] = acc / (1 - beta ** (i + 1))
    return out


def stack_seeds(curves: list[np.ndarray]) -> np.ndarray:
    """Stack per-seed curves, truncated to the shortest (runs of one config share a step count)."""
    n = min(len(c) for c in curves)
    return np.stack([c[:n] for c in curves])


def style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=3)
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def plot(path: Path, out: Path, width: float, height: float, smooth: float) -> list[Path]:
    runs = load_runs(path)
    if not runs:
        raise SystemExit(f"{path}: no runs")
    groups, varying = group_runs(runs)
    if len(groups) > len(COLORS):
        raise SystemExit(f"{len(groups)} configurations; at most {len(COLORS)} fit on one plot -- split the file")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,  # embed TrueType so text stays editable/searchable
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    fig, (ax_eval, ax_train) = plt.subplots(1, 2, figsize=(width, height), constrained_layout=True)

    train_min = np.inf
    for i, (config, members) in enumerate(groups):
        color, marker = COLORS[i], MARKERS[i]
        label = label_for(config, members, varying)

        steps = np.array([e["step"] for e in members[0]["eval_loss"]])
        evals = stack_seeds([np.array([e["loss"] for e in r["eval_loss"]]) for r in members])
        steps = steps[: evals.shape[1]]
        ax_eval.plot(steps, evals.mean(0), color=color, lw=1.5, marker=marker, ms=3.5,
                     mec="white", mew=0.5, label=label, zorder=3)
        if len(members) > 1:
            ax_eval.fill_between(steps, evals.min(0), evals.max(0), color=color, alpha=0.18, lw=0)

        train = stack_seeds([ema(np.asarray(r["train_loss"], dtype=float), smooth) for r in members])
        train_min = min(train_min, train.min())
        x = np.arange(1, train.shape[1] + 1)
        ax_train.plot(x, train.mean(0), color=color, lw=1.2, label=label, zorder=3)
        if len(members) > 1:
            ax_train.fill_between(x, train.min(0), train.max(0), color=color, alpha=0.18, lw=0)

    unit = runs[0].get("loss_unit", "nats/base")

    # (a) eval loss.
    ax_eval.set_xlabel("Training step")
    ax_eval.set_ylabel(f"Eval loss ({unit})")
    ax_eval.set_title("(a) Eval loss", loc="left", color=INK)

    # (b) training loss against the uniform level.
    uniforms = {round(r["uniform_loss"], 9) for r in runs}
    # Clip the init transient (loss well above uniform) so the converged region is readable.
    lo, hi = train_min, max(uniforms)
    ax_train.set_ylim(lo - 0.05 * (hi - lo), hi + 0.25 * (hi - lo))
    if len(uniforms) == 1:
        y = hi
        ax_train.axhline(y, color=MUTED, lw=0.8, ls=":", zorder=2)
        ax_train.annotate("uniform", (1, y), xycoords=("axes fraction", "data"), xytext=(0, 2),
                          textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=7)
    ax_train.set_xlabel("Training step")
    ax_train.set_ylabel(f"Training loss ({unit})")
    ax_train.set_title("(b) Training loss" + (f" (EMA {smooth:g})" if smooth > 0 else ""),
                       loc="left", color=INK)

    for ax in (ax_eval, ax_train):
        style_axes(ax)
        ax.set_xlim(left=0)

    handles, labels = ax_eval.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1 if width < 5 else min(len(labels), 4), frameon=False,
               handlelength=2.2, columnspacing=1.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        target = out.with_suffix(suffix)
        fig.savefig(target, bbox_inches="tight", pad_inches=0.02)
        written.append(target)
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python3 -m dna.plot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="JSONL file written by dna.train")
    p.add_argument("--out", type=Path, default=None,
                   help="output path; .pdf and .png are both written (default: next to the results file)")
    p.add_argument("--width", type=float, default=6.75, help="figure width in inches (6.75 = two-column page)")
    p.add_argument("--height", type=float, default=2.6, help="figure height in inches")
    p.add_argument("--smooth", type=float, default=0.9, help="EMA factor for training loss (0 = raw)")
    args = p.parse_args(argv)

    out = args.out or args.results.with_suffix(".pdf")
    for path in plot(args.results, out, args.width, args.height, args.smooth):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
