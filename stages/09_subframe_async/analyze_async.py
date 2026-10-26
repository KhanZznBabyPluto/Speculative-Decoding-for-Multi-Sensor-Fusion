"""
Analyse the stage-09 async-camera array (0 GPU): one NuScenes load for everything.

For every dump under dumps/async_*:
  * pooled mAP on all tokens and on STALE-ONLY tokens (drop samples whose
    realised ages are all 0 — scene starts, where no older sweep exists);
  * per-scene mAP (23 scenes, the independent unit) for paired tests.
Then:
  * uniform:J  honest vs naive  -> share of the gap-to-fresh the honest lift closes,
    as a function of real delay (the stage-04 figure was 16.7% at 500 ms);
  * rr:P vs whole:P (equal compute, equal mean age) -> does the age PATTERN matter?

Reuses stage 04's SubsetEvaluator / bootstrap statistics so numbers are
comparable with every earlier table.

Usage:
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/09_subframe_async/analyze_async.py
"""

import argparse
import json
import os
import sys

import numpy as np

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(STAGE_DIR, "dumps")
sys.path.insert(0, os.path.join(STAGE_DIR, "..", "04_streampetr_latent"))
from eval_subset import SubsetEvaluator, load_infos, load_results, token_to_scene  # noqa: E402
from bootstrap_ci import paired_tests  # noqa: E402

UNIFORM_J = (1, 2, 3, 4, 6)
FRESH = "async_uniform0_naive"
# (config, baseline) pairs for scene-level paired tests.
PAIRS = ([(f"async_uniform{j}_honest", f"async_uniform{j}_naive") for j in UNIFORM_J]
         + [("async_rr2_honest", "async_whole2_honest"),
            ("async_rr3_honest", "async_whole3_honest"),
            ("async_rr2_honest", "async_rr2_naive"),
            ("async_rr3_honest", "async_rr3_naive")])


def load_dump(name):
    """(results, meta, realised-age dict token -> 6 ages in ms) for one dump dir."""
    d = os.path.join(DUMPS, name)
    results, meta = load_results(os.path.join(d, "pts_bbox", "results_nusc.json"))
    ages_path = os.path.join(d, "realised_ages.json")
    ages = json.load(open(ages_path))["per_token_ms"] if os.path.isfile(ages_path) else {}
    return results, meta, ages


def per_scene(ev, results, meta, tok2scene, scenes):
    """(N_scenes,) per-scene mAP; nan where a scene has no tokens."""
    out = np.full(len(scenes), np.nan)
    for i, s in enumerate(scenes):
        toks = [t for t in results if tok2scene.get(t) == s]
        if toks:
            m = ev.evaluate(results, tokens=toks, meta=meta)
            out[i] = m["mean_ap"] if m is not None else np.nan
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="stage 09 analysis")
    ap.add_argument("--out", default=os.path.join(DUMPS, "analysis_async.json"))
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(DUMPS)
                   if n.startswith("async_") and os.path.isdir(os.path.join(DUMPS, n)))
    by_scene = load_infos()
    tok2scene = token_to_scene(by_scene)
    scenes = sorted(by_scene)
    ev = SubsetEvaluator()

    rows, scene_map = {}, {}
    for n, name in enumerate(names):
        results, meta, ages = load_dump(name)
        stale = [t for t in results if any(a > 0 for a in ages.get(t, [0]))]
        all_m = ev.evaluate(results, tokens=list(results), meta=meta)["mean_ap"]
        stale_m = ev.evaluate(results, tokens=stale, meta=meta)["mean_ap"] if stale else float("nan")
        mean_age = float(np.mean([np.mean(ages[t]) for t in stale])) if stale else 0.0
        rows[name] = dict(map_all=all_m, map_stale=stale_m, n_stale=len(stale),
                          mean_age_ms_stale=mean_age, stale_tokens=stale)
        scene_map[name] = per_scene(ev, results, meta, tok2scene, scenes)
        print(f"[{n + 1}/{len(names)}] {name:<24} mAP all={all_m:.4f} "
              f"stale-only={stale_m:.4f} (n={len(stale)}, age {mean_age:.0f} ms)", flush=True)

    # Fresh re-evaluated on each config's own stale token set, so gaps are like-for-like.
    fresh_results, fresh_meta, _ = load_dump(FRESH)
    for name, r in rows.items():
        r["fresh_on_same"] = (ev.evaluate(fresh_results, tokens=r["stale_tokens"], meta=fresh_meta)["mean_ap"]
                              if r["n_stale"] else float("nan"))

    print("\n=== uniform delay: honest lift vs naive (stale-only tokens) ===")
    print(" J  age ms  fresh   naive   honest  naive loss  honest loss  gap closed")
    gap = {}
    for j in UNIFORM_J:
        nv, hn = rows[f"async_uniform{j}_naive"], rows[f"async_uniform{j}_honest"]
        f = nv["fresh_on_same"]
        closed = (hn["map_stale"] - nv["map_stale"]) / (f - nv["map_stale"])
        gap[j] = dict(age_ms=nv["mean_age_ms_stale"], fresh=f, naive=nv["map_stale"],
                      honest=hn["map_stale"], gap_closed=closed)
        print(f" {j}  {nv['mean_age_ms_stale']:6.0f}  {f:.4f}  {nv['map_stale']:.4f}  {hn['map_stale']:.4f}"
              f"   {1 - nv['map_stale'] / f:7.1%}     {1 - hn['map_stale'] / f:7.1%}     {closed:6.1%}")

    print("\n=== scene-level paired tests (n=23) ===")
    tests = {}
    for a, b in PAIRS:
        t = paired_tests(scene_map[a], scene_map[b])
        tests[f"{a} vs {b}"] = t
        print(f" {a:<22} vs {b:<22} Δ={t['mean_delta']:+.4f} "
              f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}]  wins {t['wins']}/{t['n_scenes']}  "
              f"sign p={t['sign_p']:.3f}  {'SIG' if t['significant'] else 'n.s.'}")

    for r in rows.values():
        r.pop("stale_tokens")
    with open(args.out, "w") as fh:
        json.dump(dict(rows=rows, gap=gap, tests=tests, scenes=scenes,
                       per_scene_map={k: v.tolist() for k, v in scene_map.items()}), fh, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
