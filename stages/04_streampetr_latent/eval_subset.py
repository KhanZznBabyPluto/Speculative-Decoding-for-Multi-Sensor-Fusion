"""Re-evaluate already-saved nuScenes detection results on arbitrary sample subsets.

Both `stratify_by_egomotion.py` and `bootstrap_ci.py` need the same primitive:
given a `results_nusc.json` produced by an earlier eval run, plus a subset of
sample tokens, compute nuScenes-devkit mAP/NDS *on that subset only* — with no
GPU and no re-run of inference. Every config under `dumps/stale_map_*` already
has its per-sample results on disk, so the entire mitigation table can be
re-analysed for free.

Why this module exists (rather than a helper inside each script): two costs
dominate and must be amortised across many subset evals.

  - `NuScenes(version='v1.0-trainval')` metadata load  (~30 s)
  - `load_gt()` over the full val split                (~30 s)

Both are done once per process and cached; each subsequent subset eval is then
seconds. That is what makes a 23-scene x N-config sweep (and a 1000-resample
bootstrap on top of it) viable on CPU.

A note on subset mAP, which matters for how these numbers may be read:
nuScenes averages AP over all 10 classes. A subset that contains no instances
of some class scores AP=0 for it, dragging that subset's absolute mAP down.
So absolute mAP is NOT comparable ACROSS subsets with different class mixes.
It IS comparable WITHIN a subset across configs (same tokens, same class mix),
which is why the stratification script reports the *retention ratio*
mAP(delay=k) / mAP(delay=0) per stratum rather than raw mAP.

Correctness gate: evaluating any config on its FULL token set must reproduce
the mAP already recorded in that config's `metrics_summary.json`.
`python eval_subset.py --self-test` checks exactly that.

Usage (as a library):

    ev = SubsetEvaluator(DATAROOT)
    res = load_results("dumps/stale_map_delay0/pts_bbox/results_nusc.json")
    m = ev.evaluate(res, tokens=my_token_subset)
    print(m["mean_ap"], m["nd_score"])
"""

import argparse
import copy
import json
import os
import tempfile

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────

# The data/nuscenes symlink inside StreamPETR/ points at poc/data_full, which
# no longer exists; the live metadata tree is here. Passed explicitly so this
# script does not depend on that dangling symlink.
DATAROOT = "/ibex/user/khangea/multisensor/data_full"
VERSION = "v1.0-trainval"
EVAL_SET = "val"
EVAL_VERSION = "detection_cvpr_2019"

VAL_PKL = "/ibex/user/khangea/multisensor/data_full/nuscenes2d_temporal_infos_val.pkl"
DUMPS = "/ibex/user/khangea/multisensor/stages/04_streampetr_latent/dumps"

NUSC_DT = 0.5  # nuScenes keyframe spacing (2 Hz), seconds


# ── Results / metadata IO ─────────────────────────────────────────────────

def load_results(path):
    """Load a results_nusc.json.

    Returns:
        (results, meta) where results maps sample_token -> list of box dicts,
        and meta is the submission meta block the devkit requires.
    """
    with open(path) as f:
        blob = json.load(f)
    return blob["results"], blob.get("meta", {})


def config_results_path(config_name):
    """Path to a config's saved results json.

    Accepts both dump naming schemes: the staleness sweep writes
    `stale_map_<name>` (e.g. 'delay0', 'd1_velwarp'), while the honest-protocol
    array writes `honest_<delay>_<mode>` directly (e.g. 'honest_d1_key').
    """
    # An absolute path to a dump dir is also accepted, so results from other
    # stages (e.g. stage 07's age-port evals) can be bootstrapped with the same
    # machinery rather than being copied into this stage's dumps/.
    if os.path.isabs(config_name):
        p = os.path.join(config_name, "pts_bbox", "results_nusc.json")
        if os.path.isfile(p):
            return p
    for d in (f"stale_map_{config_name}", config_name):
        p = os.path.join(DUMPS, d, "pts_bbox", "results_nusc.json")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no results_nusc.json for config '{config_name}'")


def config_recorded_map(config_name):
    """The mAP already recorded on disk for a config (for the self-test)."""
    p = os.path.join(DUMPS, f"stale_map_{config_name}", "pts_bbox",
                     "metrics_summary.json")
    with open(p) as f:
        return json.load(f)["mean_ap"]


# ── Scene / ego-motion bookkeeping from the val pkl ───────────────────────

def load_infos(val_pkl=VAL_PKL):
    """Load val pkl infos, grouped by scene and sorted by timestamp.

    Returns:
        dict scene_token -> list of info dicts in temporal order.
    """
    import pickle
    with open(val_pkl, "rb") as f:
        data = pickle.load(f)
    by_scene = {}
    for info in data["infos"]:
        by_scene.setdefault(info["scene_token"], []).append(info)
    for fs in by_scene.values():
        fs.sort(key=lambda x: x["timestamp"])
    return by_scene


def token_to_scene(by_scene):
    """Map sample_token -> scene_token."""
    return {i["token"]: st for st, fs in by_scene.items() for i in fs}


def frame_index_in_scene(by_scene):
    """Map sample_token -> its 0-based index within its scene (temporal order).

    Used to identify the first k frames of each scene, where FeatureSubstitutor
    clamps the applied delay to the cache length — those frames are effectively
    FRESH even in a delay=k run, so they must be excluded from stale strata.
    """
    return {i["token"]: n for fs in by_scene.values() for n, i in enumerate(fs)}


def ego_displacement(by_scene, k):
    """Ego translation magnitude between frame f-k and frame f, per token.

    This is the quantity the 'coordinate lie' hypothesis predicts should drive
    the collapse: under a delay of k, photons captured at f-k are lifted as if
    captured at f, so the induced 3D error scales with how far the ego moved in
    between.

    Args:
        by_scene: output of load_infos().
        k: staleness in frames.

    Returns:
        dict sample_token -> float metres. Tokens in the first k frames of a
        scene are omitted (no frame f-k exists within the scene).
    """
    disp = {}
    for fs in by_scene.values():
        for n in range(k, len(fs)):
            t_now = np.asarray(fs[n]["ego2global_translation"], dtype=np.float64)
            t_old = np.asarray(fs[n - k]["ego2global_translation"], dtype=np.float64)
            disp[fs[n]["token"]] = float(np.linalg.norm(t_now - t_old))
    return disp


# ── Subset evaluation ─────────────────────────────────────────────────────

class SubsetEvaluator:
    """Runs nuScenes-devkit detection eval restricted to a set of sample tokens.

    Caches the NuScenes metadata object and the full GT EvalBoxes so that many
    subset evals in one process stay cheap. Installs a `load_gt` monkey-patch
    (the same trick measure_stale_map.py uses for the partial blob) that returns
    only the currently-allowed tokens; NuScenesEval asserts pred and gt token
    sets are equal, so predictions are filtered to the identical set.
    """

    def __init__(self, dataroot=DATAROOT, version=VERSION, eval_set=EVAL_SET,
                 verbose=True):
        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory

        if verbose:
            print(f"[eval_subset] loading NuScenes {version} metadata "
                  f"(one-time, ~30 s) ...", flush=True)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.cfg = config_factory(EVAL_VERSION)
        self.eval_set = eval_set
        self._gt_all = None      # full-split GT EvalBoxes, loaded on first use
        self._allowed = set()    # tokens the patched load_gt will return
        self._install_gt_patch()
        if verbose:
            print("[eval_subset] ready", flush=True)

    def _install_gt_patch(self):
        """Patch devkit load_gt to serve cached, token-filtered GT."""
        import nuscenes.eval.common.loaders as loaders_mod
        import nuscenes.eval.detection.evaluate as eval_mod
        from nuscenes.eval.common.data_classes import EvalBoxes

        orig_load_gt = loaders_mod.load_gt

        def patched_load_gt(nusc, eval_split, box_cls, verbose=False):
            if self._gt_all is None:
                self._gt_all = orig_load_gt(nusc, eval_split, box_cls,
                                            verbose=False)
            out = EvalBoxes()
            for tok in self._allowed:
                if tok in self._gt_all.boxes:
                    # Boxes are reused by reference across subset evals. The
                    # devkit only annotates them with deterministic derived
                    # fields (ego_translation), so sharing is safe here.
                    out.add_boxes(tok, self._gt_all.boxes[tok])
            return out

        loaders_mod.load_gt = patched_load_gt
        if hasattr(eval_mod, "load_gt"):
            eval_mod.load_gt = patched_load_gt

    def gt_tokens(self):
        """All sample tokens the full-split GT knows about (loads GT if needed)."""
        if self._gt_all is None:
            # Trigger one cheap load through the patch.
            self._allowed = set()
            import nuscenes.eval.common.loaders as loaders_mod
            from nuscenes.eval.detection.data_classes import DetectionBox
            loaders_mod.load_gt(self.nusc, self.eval_set, DetectionBox,
                                verbose=False)
        return set(self._gt_all.boxes.keys())

    def evaluate(self, results, tokens, meta=None, verbose=False):
        """Compute devkit metrics for `results` restricted to `tokens`.

        Args:
            results: dict sample_token -> list of box dicts (from load_results).
            tokens: iterable of sample tokens to evaluate on.
            meta: submission meta block; a permissive default is used if None.
            verbose: pass through to NuScenesEval.

        Returns:
            The devkit metrics_summary dict (mean_ap, nd_score, tp_errors,
            label_aps, ...), or None if the subset has no evaluable tokens.
        """
        from nuscenes.eval.detection.evaluate import NuScenesEval

        want = set(tokens) & set(results.keys()) & self.gt_tokens()
        if not want:
            return None

        self._allowed = want
        sub = {tok: results[tok] for tok in want}
        blob = {
            "meta": meta or {"use_camera": True, "use_lidar": False,
                             "use_radar": False, "use_map": False,
                             "use_external": False},
            "results": sub,
        }

        with tempfile.TemporaryDirectory() as tmp:
            res_path = os.path.join(tmp, "results_nusc.json")
            with open(res_path, "w") as f:
                json.dump(blob, f)
            nusc_eval = NuScenesEval(
                self.nusc, config=self.cfg, result_path=res_path,
                eval_set=self.eval_set, output_dir=tmp, verbose=verbose,
            )
            nusc_eval.main(render_curves=False)
            with open(os.path.join(tmp, "metrics_summary.json")) as f:
                return json.load(f)


# ── Self-test ─────────────────────────────────────────────────────────────

def run_self_test(configs=("delay0", "delay1", "d1_velwarp")):
    """Full-token-set eval must reproduce each config's recorded mAP.

    This is the gate on the whole approach: if the plumbing here disagreed with
    the numbers already in the paper, every stratified/bootstrapped result
    downstream would be untrustworthy.
    """
    ev = SubsetEvaluator()
    ok = True
    print(f"\n{'config':<16} {'recorded':>10} {'recomputed':>12} {'delta':>10}")
    print("-" * 52)
    for name in configs:
        results, meta = load_results(config_results_path(name))
        m = ev.evaluate(results, tokens=results.keys(), meta=meta)
        recorded = config_recorded_map(name)
        delta = m["mean_ap"] - recorded
        flag = "OK" if abs(delta) < 1e-4 else "MISMATCH"
        if abs(delta) >= 1e-4:
            ok = False
        print(f"{name:<16} {recorded:>10.4f} {m['mean_ap']:>12.4f} "
              f"{delta:>+10.6f}  {flag}")
    print("-" * 52)
    print("SELF-TEST PASS" if ok else "SELF-TEST FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="verify full-set eval reproduces recorded mAP")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(0 if run_self_test() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
