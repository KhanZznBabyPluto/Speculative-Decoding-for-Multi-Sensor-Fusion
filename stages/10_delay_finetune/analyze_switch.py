"""
Stage 10d analysis: mode switching under a frame-to-frame varying delay (0 GPU).

Inputs are the runs from submit_switch.sh for one schedule (pretrained only,
delay-FT only, real in-loop switch at >0 ms and, if present, >100 ms). Adds a SPLICE per threshold:
each frame's detections taken from whichever single-model run the rule picks,
so every model kept its own memory. splice − switch = the price of each model
reading temporal memory written by the other.

Reports pooled mAP on all / fresh-only / stale-only frames, scene-level paired
tests (n=23) for the pre-registered comparisons, and the same switch-vs-single
tests restricted to fresh and to stale frames.

Usage:
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/10_delay_finetune/analyze_switch.py --schedule rand3
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_DIR = "/ibex/user/khangea/multisensor"
sys.path.insert(0, os.path.join(PROJECT_DIR, "stages/04_streampetr_latent"))
from bootstrap_ci import paired_tests, per_scene_map  # noqa: E402
from eval_subset import SubsetEvaluator, load_infos, load_results, token_to_scene  # noqa: E402

DUMPS = os.path.join(PROJECT_DIR, "stages/10_delay_finetune/dumps")
THRESHOLDS = {"0": 0.0, "100": 100.0}
PREFIX = None   # set in main from --schedule


def dump_dir(name):
    return os.path.join(DUMPS, f"{PREFIX}_{name}")


def per_scene_subset(ev, d, tok2scene, scenes, keep):
    """Per-scene mAP of dump `d` restricted to tokens in `keep` (nan if none)."""
    res, meta = load_results(os.path.join(d, "pts_bbox", "results_nusc.json"))
    out = []
    for s in scenes:
        toks = [t for t in res if tok2scene.get(t) == s and t in keep]
        m = ev.evaluate(res, toks, meta=meta) if toks else None
        out.append(m["mean_ap"] if m else np.nan)
    return np.array(out)


def write_splice(name, ages_ms, above_ms):
    """Build the splice dump for one threshold; returns its dir (skips if it exists)."""
    out = dump_dir(f"splice{name}")
    path = os.path.join(out, "pts_bbox", "results_nusc.json")
    if os.path.isfile(path):
        return out
    pre, meta = load_results(os.path.join(dump_dir("pretrained"), "pts_bbox", "results_nusc.json"))
    dft, _ = load_results(os.path.join(dump_dir("delayft"), "pts_bbox", "results_nusc.json"))
    spliced = {tok: (dft if max(ages_ms[tok]) > above_ms else pre)[tok] for tok in pre}
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": spliced}, f)
    return out


def main():
    global PREFIX
    ap = argparse.ArgumentParser(description="stage 10d mode-switch analysis")
    ap.add_argument("--schedule", default="rand3", help="schedule tag in dump names, e.g. rand3, rand1")
    args = ap.parse_args()
    PREFIX = f"switch_{args.schedule}_honest"
    out_json = os.path.join(DUMPS, f"analysis_switch_{args.schedule}.json")

    ages_ms = json.load(open(os.path.join(dump_dir("switch0"), "realised_ages.json")))["per_token_ms"]
    fresh = {t for t, a in ages_ms.items() if max(a) == 0}
    stale = set(ages_ms) - fresh
    runs = {n: dump_dir(n) for n in ("pretrained", "delayft", "switch0", "switch100")
            if os.path.isdir(dump_dir(n))}
    for name, thr in THRESHOLDS.items():
        if f"switch{name}" in runs:
            runs[f"splice{name}"] = write_splice(name, ages_ms, thr)

    by_scene = load_infos()
    tok2scene = token_to_scene(by_scene)
    scenes = sorted(by_scene)
    ev = SubsetEvaluator()

    pooled, per_scene = {}, {}
    for n, d in runs.items():
        res, meta = load_results(os.path.join(d, "pts_bbox", "results_nusc.json"))
        pooled[n] = {part: ev.evaluate(res, toks, meta=meta)["mean_ap"]
                     for part, toks in (("all", set(res)), ("fresh", fresh), ("stale", stale))}
        per_scene[n] = per_scene_map(ev, d, tok2scene, scenes)
        print(f"{n:11s} all {pooled[n]['all']:.4f}  fresh {pooled[n]['fresh']:.4f}  "
              f"stale {pooled[n]['stale']:.4f}", flush=True)

    pairs = [("switch0", "pretrained"), ("switch0", "delayft"),       # pre-registered
             ("switch100", "pretrained"), ("switch100", "delayft"),
             ("splice0", "switch0"), ("splice100", "switch100"),      # cost of cross-model memory
             ("splice0", "delayft"), ("delayft", "pretrained")]
    tests = {f"{a}_vs_{b}": paired_tests(per_scene[a], per_scene[b])
             for a, b in pairs if a in runs and b in runs}
    for part, keep in (("fresh", fresh), ("stale", stale)):
        sub = {n: per_scene_subset(ev, runs[n], tok2scene, scenes, keep)
               for n in ("pretrained", "delayft", "switch0")}
        for b in ("delayft", "pretrained"):
            tests[f"{part}:switch0_vs_{b}"] = paired_tests(sub["switch0"], sub[b])
    print(f"\nfresh frames {len(fresh)}, stale frames {len(stale)}")
    for k, t in tests.items():
        print(f"  {k:24s} {t['mean_delta']:+.4f} [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] "
              f"{t['wins']:>2}/{t['n_scenes']}  p={t['sign_p']:.3f}  "
              f"{'SIG' if t['significant'] else 'n.s.'}")

    with open(out_json, "w") as f:
        json.dump(dict(pooled=pooled, n_fresh=len(fresh), n_stale=len(stale),
                       per_scene={n: v.tolist() for n, v in per_scene.items()},
                       tests=tests), f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
