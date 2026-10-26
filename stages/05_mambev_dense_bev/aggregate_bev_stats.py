"""Stage 05 — Scalable AI deliverable, aggregator + plot generator.

Reads all per-scene JSONLs produced by run_bev_stats.py + the SLURM array
in submit_bev_stats.sh, computes distribution stats, and writes:

  - dumps/scale/aggregated.csv           ← all rows merged
  - dumps/scale/summary.txt              ← headline numerical claims
  - dumps/scale/plot_cos_histogram.png   ← H1: cosine distribution
  - dumps/scale/plot_cos_vs_speed.png    ← H3: cos vs ego speed
  - dumps/scale/plot_warp_vs_motion.png  ← H2: warp gain vs ego motion magnitude
  - dumps/scale/plot_per_scene.png       ← per-scene cos distribution

The per-row schema is whatever run_bev_stats.py wrote — see that module's
`row = dict(...)` for the field list.

Usage:
  python aggregate_bev_stats.py [--in-dir dumps/scale] [--out-dir dumps/scale]

The warp_gain column is NOT present in run_bev_stats.py output (computing
it would require re-running the warp). For Scalable AI deliverable we
have two paths:
  (a) accept H1 + H3 from this script's output (cos distribution);
  (b) for H2, run warp_retention_test.py on a subset of dumps as a
      separate validation step. The 3-scene Phase 2.1 result already
      establishes the direction; this script proves universality of H1.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

DEFAULT_DIR = Path("/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/scale")


def load_jsonls(in_dir: Path):
    rows = []
    files = sorted(in_dir.glob("scene_*.jsonl"))
    print(f"Reading {len(files)} per-scene JSONLs from {in_dir}")
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows total.")
    return rows


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(rows, out_path: Path):
    cos_means       = [r["cos_mean"]      for r in rows]
    cos_medians     = [r["cos_median"]    for r in rows]
    l2_means        = [r["l2_mean"]       for r in rows]
    frac_cos_85     = [r["frac_cos_gt_85"] for r in rows]
    egos_trans      = [r["ego_trans"]     for r in rows]
    egos_dyaw       = [r["ego_dyaw"]      for r in rows]
    top1_t          = [r["top1_score_t"]  for r in rows]
    top1_tk         = [r["top1_score_tk"] for r in rows]
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r["cos_mean"])

    H1_pass = sum(1 for c in cos_means if c > 0.85) / max(1, len(cos_means))

    lines = []
    lines.append(f"Stage 05 Scalable-AI aggregate")
    lines.append(f"  rows      : {len(rows)}")
    lines.append(f"  scenes    : {len(by_scene)}")
    lines.append("")
    lines.append("Per-pair BEV cosine:")
    lines.append(f"  mean      : {sum(cos_means)/len(cos_means):.4f}")
    lines.append(f"  median    : {percentile(cos_means, 0.5):.4f}")
    lines.append(f"  p05 / p95 : {percentile(cos_means, 0.05):.4f} / {percentile(cos_means, 0.95):.4f}")
    lines.append(f"  min / max : {min(cos_means):.4f} / {max(cos_means):.4f}")
    lines.append("")
    lines.append("Per-pair BEV L2:")
    lines.append(f"  mean      : {sum(l2_means)/len(l2_means):.4f}")
    lines.append(f"  median    : {percentile(l2_means, 0.5):.4f}")
    lines.append(f"  p05 / p95 : {percentile(l2_means, 0.05):.4f} / {percentile(l2_means, 0.95):.4f}")
    lines.append("")
    lines.append("Ego motion (per-pair):")
    lines.append(f"  ‖Δ‖ mean / p95 / max  : "
                 f"{sum(egos_trans)/len(egos_trans):.3f} m / "
                 f"{percentile(egos_trans, 0.95):.3f} m / "
                 f"{max(egos_trans):.3f} m")
    lines.append(f"  |Δyaw| mean / p95 / max : "
                 f"{sum(abs(y) for y in egos_dyaw)/len(egos_dyaw):.3f}° / "
                 f"{percentile([abs(y) for y in egos_dyaw], 0.95):.3f}° / "
                 f"{max(abs(y) for y in egos_dyaw):.3f}°")
    lines.append("")
    lines.append("Top-1 detection score (per-pair):")
    lines.append(f"  anchor t  mean : {sum(top1_t)/len(top1_t):.3f}")
    lines.append(f"  anchor tk mean : {sum(top1_tk)/len(top1_tk):.3f}")
    lines.append("")
    lines.append("Hypothesis tests:")
    lines.append(f"  H1: P(cos > 0.85)             = {H1_pass:.4f}   (claim: > 0.95)")
    lines.append(f"  H1 strict (cos > 0.9)         = {sum(1 for c in cos_means if c > 0.9)/len(cos_means):.4f}")
    lines.append(f"  H1 very strict (cos > 0.95)   = {sum(1 for c in cos_means if c > 0.95)/len(cos_means):.4f}")
    lines.append("")
    lines.append("Per-scene cosine summary:")
    for scene in sorted(by_scene.keys()):
        v = by_scene[scene]
        lines.append(f"  {scene[:8]}: n={len(v):3d}  mean={sum(v)/len(v):.4f}  "
                     f"min={min(v):.4f}  max={max(v):.4f}")

    summary = "\n".join(lines)
    out_path.write_text(summary + "\n")
    print(summary)


def to_csv(rows, out_path: Path):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote CSV → {out_path}")


def make_plots(rows, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    cos    = [r["cos_mean"]   for r in rows]
    trans  = [r["ego_trans"]  for r in rows]
    dyaw   = [r["ego_dyaw"]   for r in rows]
    scenes = [r["scene"][:8]  for r in rows]
    l2     = [r["l2_mean"]    for r in rows]

    # H1: cosine histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cos, bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(0.85, color="red", linestyle="--", label="claim threshold 0.85")
    ax.set_xlabel("mean per-cell cosine(BEV_t, BEV_{t+k})")
    ax.set_ylabel("# frame pairs")
    ax.set_title(f"BEV cosine distribution (N={len(rows)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cos_histogram.png", dpi=140)
    plt.close(fig)

    # H3: cos vs ego speed
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(trans, cos, s=10, alpha=0.5)
    ax.set_xlabel("ego translation ‖Δ‖ (m, per 0.5 s)")
    ax.set_ylabel("mean per-cell cosine")
    ax.set_title("BEV cosine vs ego motion magnitude")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_cos_vs_speed.png", dpi=140)
    plt.close(fig)

    # L2 vs ego motion
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(trans, l2, s=10, alpha=0.5)
    ax.set_xlabel("ego translation ‖Δ‖ (m, per 0.5 s)")
    ax.set_ylabel("mean per-cell L2")
    ax.set_title("BEV L2 vs ego motion magnitude")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_l2_vs_speed.png", dpi=140)
    plt.close(fig)

    # Per-scene boxplot
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"][:8]].append(r["cos_mean"])
    keys = sorted(by_scene.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([by_scene[k] for k in keys], labels=keys)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5)
    ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("per-pair cosine")
    ax.set_title("Per-scene BEV cosine distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_per_scene.png", dpi=140)
    plt.close(fig)

    print(f"Wrote 4 plots to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir",  type=Path, default=DEFAULT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonls(args.in_dir)
    if not rows:
        print("No rows to aggregate; bailing.")
        return 1
    to_csv(rows, args.out_dir / "aggregated.csv")
    summarize(rows, args.out_dir / "summary.txt")
    make_plots(rows, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
