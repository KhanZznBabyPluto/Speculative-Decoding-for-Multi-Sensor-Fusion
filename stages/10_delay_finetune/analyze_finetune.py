"""
Scene-level paired comparison of the stage-10 fine-tunes (0 GPU).

Per-scene mAP (23 val scenes, the independent unit) for 3 models x 8 delay
conditions, computed once in one process (NuScenes metadata loads once), then:
  * PRIMARY  delay-FT vs control-FT per condition (pre-registered: honest
             76-250 ms must improve; fresh must not drop by more than 0.005)
  * control-FT vs pretrained, delay-FT vs pretrained (what fine-tuning on the
             62 local train scenes does by itself)

Usage:
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/10_delay_finetune/analyze_finetune.py
"""

import json
import os
import sys

import numpy as np

PROJECT_DIR = "/ibex/user/khangea/multisensor"
sys.path.insert(0, os.path.join(PROJECT_DIR, "stages/04_streampetr_latent"))
from bootstrap_ci import paired_tests, per_scene_map  # noqa: E402
from eval_subset import SubsetEvaluator, load_infos, token_to_scene  # noqa: E402

S09 = os.path.join(PROJECT_DIR, "stages/09_subframe_async/dumps")
S10 = os.path.join(PROJECT_DIR, "stages/10_delay_finetune/dumps")
OUT_JSON = os.path.join(S10, "analysis_finetune_agelr_i7380.json")   # earlier: analysis_finetune_i7380.json (10), _age_ (10b)
CONDITIONS = ["uniform0_naive", "uniform1_naive", "uniform1_honest", "uniform2_naive",
              "uniform2_honest", "uniform3_naive", "uniform3_honest", "rr2_honest"]
MODELS = {
    "pretrained": lambda c: os.path.join(S09, f"async_{c}"),
    "control":    lambda c: os.path.join(S10, f"eval_ft_control_p0_i7380_{c}"),
    "delay":      lambda c: os.path.join(S10, f"eval_ft_delay_p0.5_i7380_{c}"),
    "age":        lambda c: os.path.join(S10, f"eval_ft_age_delay_p0.5_i7380_{c}"),   # stage 10b
    **{f"age_lrx{m}": (lambda c, m=m: os.path.join(S10, f"eval_ft_age_delay_p0.5_lrx{m}_i7380_{c}"))
       for m in (10, 30, 100)},                                                    # stage 10c
}
COMPARISONS = [("delay", "control"), ("control", "pretrained"), ("delay", "pretrained"),
               ("age", "control"), ("age", "delay")]   # stage 10b pre-registered comparisons
COMPARISONS += [(f"age_lrx{m}", ref) for m in (10, 30, 100) for ref in ("control", "delay")]   # stage 10c


def main():
    by_scene = load_infos()
    tok2scene = token_to_scene(by_scene)
    scenes = sorted(by_scene)
    ev = SubsetEvaluator()

    per_scene = {m: {} for m in MODELS}
    for m, path_of in MODELS.items():
        for c in CONDITIONS:
            print(f"\n=== {m} / {c} ===", flush=True)
            per_scene[m][c] = per_scene_map(ev, path_of(c), tok2scene, scenes)

    tests = {f"{a}_vs_{b}": {c: paired_tests(per_scene[a][c], per_scene[b][c]) for c in CONDITIONS}
             for a, b in COMPARISONS}

    print("\n" + "=" * 96)
    for name, per_c in tests.items():
        print(f"\n{name}  (mean per-scene delta, 95% CI, wins/23, sign p)")
        for c, t in per_c.items():
            print(f"  {c:17s} {t['mean_delta']:+.4f} [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] "
                  f"{t['wins']:>2}/{t['n_scenes']}  p={t['sign_p']:.3f}  "
                  f"{'SIG' if t['significant'] else 'n.s.'}")

    with open(OUT_JSON, "w") as f:
        json.dump(dict(scenes=scenes, conditions=CONDITIONS,
                       per_scene_map={m: {c: v.tolist() for c, v in d.items()} for m, d in per_scene.items()},
                       tests=tests), f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
