"""Does the staleness collapse scale with EGO motion? (zero GPU, no inference)

The hypothesis under test — call it the 'coordinate lie'. Our staleness harness
substitutes only the post-FPN image features from frame t-k; every piece of
metadata stays at frame t. Concretely, verified in source:

  - measure_stale_map.py `FeatureSubstitutor._hook` patches only
    `Petr3D.extract_img_feat`; data['timestamp'], data['ego_pose'] and
    lidar2img all come from the CURRENT sample.
  - streampetr_head.py:402  `img2lidars = data['lidar2img'].inverse()`
    lifts the stale pixels into the ego frame at t-k, but the head reads the
    result as ego-frame-at-t coordinates.
  - streampetr_head.py:426  `rec_ego_pose = torch.eye(4)`   (unconditional)
  - streampetr_head.py:438  `time_embedding(pos2posemb1d(zeros))` (unconditional)

So a query born from stale photons is told "you are here, now". The induced 3D
error is the ego displacement over k*0.5 s — mean 3.09 m at k=1 on this subset.
If that is what drives the measured collapse (0.445 -> 0.133 mAP at k=1), then
degradation must be much milder on frames where the ego barely moved, and
severe where it moved a lot. If instead near-static frames collapse just as
hard, ego motion is NOT the story and the staleness result stands as measured.

This is the cheap decisive screen: it re-uses the per-sample detections already
saved under dumps/stale_map_*/pts_bbox/results_nusc.json, so it costs no GPU
and no inference — only devkit re-evaluation on token subsets.

Two details that matter for correctness:

  - The first k frames of every scene fall back to FRESH features (the
    substitutor clamps the delay to the cache length), so they are excluded
    from the stale strata; leaving them in would dilute the collapse.
  - Absolute mAP is not comparable across strata (nuScenes averages AP over all
    10 classes, and strata have different class mixes). Within a stratum the
    token set is identical across configs, so the RETENTION RATIO
    mAP(delay=k)/mAP(delay=0) is the comparable quantity, and it is what the
    hypothesis actually speaks to.

Usage:

    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/04_streampetr_latent/stratify_by_egomotion.py --delay 1
    python stages/04_streampetr_latent/stratify_by_egomotion.py --delay 1 \
        --configs delay1 d1_velwarp d1_zero --bins fixed
"""

import argparse
import json
import os

import numpy as np

from eval_subset import (DUMPS, NUSC_DT, SubsetEvaluator, config_results_path,
                         ego_displacement, frame_index_in_scene, load_infos,
                         load_results, token_to_scene)

# ── Config ────────────────────────────────────────────────────────────────

# Physically-meaningful displacement bins (metres travelled over k*0.5 s).
# The first bin is the near-static regime where the coordinate lie is ~a no-op:
# if the collapse were purely an ego artefact, retention here should approach 1.
FIXED_BINS = [0.0, 0.5, 2.0, 4.0, np.inf]

N_QUANTILE_BINS = 4

# Classes that are physically incapable of moving. They are the discriminator
# between the two explanations of the stratified result:
#   - 'coordinate lie': stale photons lifted at the wrong ego pose -> even a
#     stationary cone lands in the wrong place, and the error grows with ego
#     displacement.
#   - 'scene dynamism confound': low ego motion just correlates with everything
#     else moving less (stopped at a light), so retention rises for boring
#     reasons.
# A cone cannot move. If cone AP still degrades with ego displacement, the
# damage is ego-induced and the confound cannot explain it.
STATIC_CLASSES = ("traffic_cone", "barrier")


# ── Strata ────────────────────────────────────────────────────────────────

def build_strata(disp, mode, n_bins=N_QUANTILE_BINS):
    """Partition tokens into ego-displacement strata.

    Args:
        disp: dict token -> ego displacement in metres over the staleness gap.
        mode: 'fixed' for FIXED_BINS, 'quantile' for equal-count bins.
        n_bins: number of quantile bins when mode='quantile'.

    Returns:
        list of (label, edge_lo, edge_hi, [tokens]) in ascending displacement.
    """
    tokens = np.array(list(disp.keys()))
    vals = np.array([disp[t] for t in tokens])

    if mode == "fixed":
        edges = np.array(FIXED_BINS)
    else:
        qs = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(vals, qs)
        edges[0], edges[-1] = -np.inf, np.inf

    strata = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (vals >= lo) & (vals < hi)
        if sel.sum() == 0:
            continue
        lo_s = "0" if not np.isfinite(lo) else f"{lo:.2f}"
        hi_s = "inf" if not np.isfinite(hi) else f"{hi:.2f}"
        strata.append((f"[{lo_s},{hi_s}) m", lo, hi, list(tokens[sel])))
    return strata


def stale_eligible_tokens(by_scene, k):
    """Tokens where a delay=k run actually applied k frames of staleness.

    Excludes the first k frames of every scene, where FeatureSubstitutor clamps
    the delay to the (shorter) cache and the frame is effectively fresh.
    """
    fidx = frame_index_in_scene(by_scene)
    return {tok for tok, n in fidx.items() if n >= k}


# ── Main ──────────────────────────────────────────────────────────────────

def run(delay, configs, bins, out_json):
    """Evaluate each config per ego-displacement stratum and report retention."""
    by_scene = load_infos()
    disp = ego_displacement(by_scene, k=delay)
    eligible = stale_eligible_tokens(by_scene, k=delay)
    disp = {t: d for t, d in disp.items() if t in eligible}

    vals = np.array(list(disp.values()))
    print(f"\nStaleness gap k={delay} ({delay * NUSC_DT:.1f} s). "
          f"Stale-eligible tokens: {len(disp)} "
          f"(excluded {len(by_scene)} x first-{delay}-frame fresh fallbacks)")
    print(f"Ego displacement over the gap: mean {vals.mean():.2f} m, "
          f"median {np.median(vals):.2f} m, min {vals.min():.2f} m, "
          f"max {vals.max():.2f} m")

    strata = build_strata(disp, bins)
    ev = SubsetEvaluator()

    # Baseline is always the fresh run, evaluated on the SAME tokens.
    all_configs = ["delay0"] + list(configs)
    results_cache = {c: load_results(config_results_path(c)) for c in all_configs}

    table = {}
    for label, lo, hi, tokens in strata:
        print(f"\n=== stratum {label}  (n={len(tokens)}) ===", flush=True)
        row = {"n": len(tokens), "lo": float(lo), "hi": float(hi)}
        for c in all_configs:
            res, meta = results_cache[c]
            m = ev.evaluate(res, tokens=tokens, meta=meta)
            if m is None:
                continue
            row[c] = {"mAP": m["mean_ap"], "NDS": m["nd_score"],
                      "mATE": m["tp_errors"]["trans_err"],
                      # Per-class AP is what disentangles the ego-lie from the
                      # scene-dynamism confound: STATIC_CLASSES cannot move, so
                      # any AP they lose under staleness is ego-induced.
                      "label_aps": {k: float(np.mean(list(v.values())))
                                    for k, v in m["label_aps"].items()}}
            print(f"  {c:<16} mAP={m['mean_ap']:.4f}  NDS={m['nd_score']:.4f}  "
                  f"mATE={m['tp_errors']['trans_err']:.3f}", flush=True)
        table[label] = row

    _print_summary(table, configs, delay)

    with open(out_json, "w") as f:
        json.dump({"delay": delay, "bins": bins, "table": table}, f, indent=2)
    print(f"\nSaved -> {out_json}")
    return table


def _print_summary(table, configs, delay):
    """Print the retention table — the actual answer to the hypothesis."""
    print("\n" + "=" * 78)
    print(f"RETENTION vs fresh, per ego-displacement stratum (delay={delay})")
    print("Retention = mAP(config) / mAP(delay0) on the SAME tokens.")
    print("Coordinate-lie prediction: retention RISES sharply as ego motion -> 0")
    print("=" * 78)

    header = f"{'stratum':<16} {'n':>5} {'fresh mAP':>10}"
    for c in configs:
        header += f" {c:>14}"
    print(header)
    print("-" * 78)

    for label, row in table.items():
        base = row.get("delay0", {}).get("mAP")
        line = f"{label:<16} {row['n']:>5} {base:>10.4f}"
        for c in configs:
            m = row.get(c, {}).get("mAP")
            if m is None or not base:
                line += f" {'--':>14}"
            else:
                line += f" {m:.4f} ({100 * m / base:4.1f}%)"
        print(line)
    print("-" * 78)
    _print_static_class_discriminator(table, configs[0])


def _print_static_class_discriminator(table, stale_config):
    """Retention of provably-stationary classes, per ego-displacement stratum.

    Cones and barriers do not move, so object motion cannot explain any AP they
    lose. If their retention still falls as ego displacement grows, the damage
    is ego-induced — which is the coordinate lie, not a dynamism confound.
    """
    print("\n" + "=" * 78)
    print(f"DISCRIMINATOR: static-class AP retention ({stale_config} vs fresh)")
    print("These objects CANNOT move. Any loss here is not object motion.")
    print("=" * 78)
    header = f"{'stratum':<16} {'n':>5}"
    for cls in STATIC_CLASSES:
        header += f" {cls:>24}"
    print(header)
    print("-" * 78)
    for label, row in table.items():
        line = f"{label:<16} {row['n']:>5}"
        for cls in STATIC_CLASSES:
            fresh = row.get("delay0", {}).get("label_aps", {}).get(cls)
            stale = row.get(stale_config, {}).get("label_aps", {}).get(cls)
            if fresh is None or stale is None or fresh < 1e-6:
                line += f" {'--':>24}"
            else:
                line += f"   {fresh:.3f}->{stale:.3f} ({100 * stale / fresh:5.1f}%)"
        print(line)
    print("-" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delay", type=int, default=1,
                    help="staleness in frames (1 = 500 ms)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="config names under dumps/stale_map_* "
                         "(default: delay{k} plus its vel_warp variant)")
    ap.add_argument("--bins", choices=["fixed", "quantile"], default="fixed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configs = args.configs or [f"delay{args.delay}", f"d{args.delay}_velwarp"]
    out = args.out or os.path.join(
        DUMPS, f"stratified_egomotion_d{args.delay}_{args.bins}.json")
    run(args.delay, configs, args.bins, out)


if __name__ == "__main__":
    main()
