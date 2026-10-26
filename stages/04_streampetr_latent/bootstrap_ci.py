"""Scene-level bootstrap CIs and paired tests for the mitigation table (0 GPU).

Every number in the paper's mitigation table is a single point estimate with no
uncertainty attached, and the two claims that carry the most weight are small:

    vel_warp   0.1378  vs  none 0.1331   (+0.005, the ONLY positive result)
    LTC v2     0.1106  vs  LTC v1 0.1081 (+0.003, the 'chimera resolved' claim)

The effective sample size is NOT 914 frames. nuScenes runs at 2 Hz, so a scene
is ~40 frames of the same road and the same objects re-observed — frames within
a scene are heavily correlated. The independent unit is the SCENE, and there are
only 23. Deltas of 0.005 may be well inside scene-to-scene variance.

This script computes, per config:
  - per-scene mAP (the devkit run once per scene, reusing saved detections)
  - a cluster bootstrap over scenes: resample the 23 scenes with replacement,
    recompute the mean-of-per-scene-mAP, repeat -> percentile CI
  - paired per-scene comparison against a baseline config (Wilcoxon signed-rank
    and an exact sign test), which is the cleanest non-parametric answer to
    "does this mitigation actually beat doing nothing?"

Why mean-of-per-scene-mAP rather than pooled mAP: a bootstrap resample repeats
scenes, and the devkit keys detections by sample token, so a pooled re-eval
cannot represent a duplicated scene. Bootstrapping the per-scene statistic is
the standard cluster-bootstrap move and needs each scene evaluated only once.
Note the mean-of-per-scene-mAP differs slightly from the pooled mAP reported in
the paper (different weighting of scenes and class mixes); the paired tests do
not depend on that identity and are the primary result here.

Usage:

    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/04_streampetr_latent/bootstrap_ci.py \
        --configs delay1 d1_velwarp d1_zero d1_freeze d1_ltc d1_ltcv2 \
        --baseline delay1
"""

import argparse
import json
import os

import numpy as np

from eval_subset import (DUMPS, SubsetEvaluator, config_results_path,
                         load_infos, load_results, token_to_scene)

# ── Config ────────────────────────────────────────────────────────────────

N_BOOTSTRAP = 10000
CI_PERCENTILES = (2.5, 97.5)
RNG_SEED = 0


# ── Per-scene evaluation ──────────────────────────────────────────────────

def per_scene_map(ev, config, tok2scene, scenes):
    """Evaluate one config once per scene.

    Args:
        ev: SubsetEvaluator.
        config: config name under dumps/stale_map_*.
        tok2scene: dict sample_token -> scene_token.
        scenes: ordered list of scene tokens.

    Returns:
        (N_scenes,) float array of per-scene mAP (nan where a scene has no
        evaluable tokens).
    """
    results, meta = load_results(config_results_path(config))
    by_scene_tokens = {s: [] for s in scenes}
    for tok in results:
        s = tok2scene.get(tok)
        if s is not None:
            by_scene_tokens[s].append(tok)

    out = np.full(len(scenes), np.nan)
    for i, s in enumerate(scenes):
        toks = by_scene_tokens[s]
        if not toks:
            continue
        m = ev.evaluate(results, tokens=toks, meta=meta)
        if m is not None:
            out[i] = m["mean_ap"]
        print(f"  {config:<14} scene {i + 1:>2}/{len(scenes)}  "
              f"n={len(toks):>3}  mAP={out[i]:.4f}", flush=True)
    return out


# ── Statistics ────────────────────────────────────────────────────────────

def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Cluster bootstrap over scenes for the mean of a per-scene statistic.

    Args:
        values: (N_scenes,) per-scene statistic, may contain nan.
        n_boot: number of resamples.

    Returns:
        (mean, lo, hi) percentile CI.
    """
    v = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, CI_PERCENTILES)
    return float(v.mean()), float(lo), float(hi)


def paired_tests(a, b, seed=RNG_SEED):
    """Paired per-scene comparison of config a against baseline b.

    Args:
        a, b: (N_scenes,) per-scene mAP arrays, aligned by scene.

    Returns:
        dict with the mean paired delta, its bootstrap CI, the number of scenes
        where a > b, an exact binomial sign-test p-value, and a Wilcoxon
        signed-rank p-value.
    """
    from scipy import stats

    ok = ~np.isnan(a) & ~np.isnan(b)
    d = a[ok] - b[ok]
    n = len(d)
    wins = int((d > 0).sum())

    mean_d, lo, hi = bootstrap_mean_ci(d, seed=seed)
    sign_p = float(stats.binomtest(wins, n, 0.5).pvalue)
    try:
        wilcoxon_p = float(stats.wilcoxon(d).pvalue)
    except ValueError:  # all-zero differences
        wilcoxon_p = 1.0

    return {"n_scenes": n, "mean_delta": mean_d, "ci_lo": lo, "ci_hi": hi,
            "wins": wins, "sign_p": sign_p, "wilcoxon_p": wilcoxon_p,
            "significant": bool(lo > 0 or hi < 0)}


# ── Main ──────────────────────────────────────────────────────────────────

def run(configs, baseline, out_json):
    """Compute per-scene mAP for each config, then CIs and paired tests."""
    by_scene = load_infos()
    tok2scene = token_to_scene(by_scene)
    scenes = sorted(by_scene.keys())
    print(f"Scenes: {len(scenes)} (the independent unit; NOT 914 frames)")

    ev = SubsetEvaluator()
    all_configs = list(dict.fromkeys([baseline] + list(configs)))

    per_scene = {}
    for c in all_configs:
        print(f"\n=== per-scene eval: {c} ===", flush=True)
        per_scene[c] = per_scene_map(ev, c, tok2scene, scenes)

    _print_report(per_scene, all_configs, baseline)

    payload = {"scenes": scenes, "baseline": baseline,
               "per_scene_map": {c: v.tolist() for c, v in per_scene.items()},
               "tests": {c: paired_tests(per_scene[c], per_scene[baseline])
                         for c in all_configs if c != baseline}}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_json}")


def _print_report(per_scene, configs, baseline):
    """Print bootstrap CIs and the paired tests against the baseline."""
    print("\n" + "=" * 84)
    print(f"BOOTSTRAP CIs over {len(per_scene[baseline])} scenes "
          f"({N_BOOTSTRAP} resamples), mean-of-per-scene mAP")
    print("=" * 84)
    print(f"{'config':<16} {'mean mAP':>10} {'95% CI':>22}")
    print("-" * 84)
    for c in configs:
        mean, lo, hi = bootstrap_mean_ci(per_scene[c])
        print(f"{c:<16} {mean:>10.4f}   [{lo:.4f}, {hi:.4f}]")
    print("-" * 84)

    print("\n" + "=" * 84)
    print(f"PAIRED per-scene comparison vs baseline '{baseline}'")
    print("A mitigation is only real if the CI on the paired delta excludes 0.")
    print("=" * 84)
    print(f"{'config':<16} {'mean d':>9} {'95% CI on delta':>22} "
          f"{'wins':>7} {'sign p':>9} {'signif':>8}")
    print("-" * 84)
    for c in configs:
        if c == baseline:
            continue
        t = paired_tests(per_scene[c], per_scene[baseline])
        print(f"{c:<16} {t['mean_delta']:>+9.4f}   "
              f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] "
              f"{t['wins']:>3}/{t['n_scenes']:<3} {t['sign_p']:>9.3f} "
              f"{'YES' if t['significant'] else 'no':>8}")
    print("-" * 84)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+",
                    default=["delay1", "d1_velwarp", "d1_zero", "d1_freeze",
                             "d1_ltc", "d1_ltcv2"])
    ap.add_argument("--baseline", default="delay1")
    ap.add_argument("--out", default=os.path.join(DUMPS, "bootstrap_ci_d1.json"))
    args = ap.parse_args()
    run(args.configs, args.baseline, args.out)


if __name__ == "__main__":
    main()
