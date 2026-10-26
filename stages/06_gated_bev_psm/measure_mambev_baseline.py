"""Establish MamBEV's OWN detection baseline (mAP/NDS) on our 23-scene val subset.

WHY THIS EXISTS
---------------
We have never measured MamBEV's detection quality. Not once, on any split.
Stage 05 verified only that (a) the checkpoint loads 1101/1105 weights, and
(b) a synthetic forward pass on random images produces 300 boxes — then went
straight to analysing the encoder's BEV tensor and concluding "the BEV is a
query-position cache, not a metric world grid".

That conclusion rests on internal representations of a model whose task
performance was never verified on our data. If MamBEV scores far below its
published 42.2 mAP / 52.5 NDS here — because of the pkl format, the R101-DCN
backbone, the frames=[-2,-1,0] temporal stack, or the partial blob — then the
cosine statistics in stage 05 were computed on a model that is not working, and
the cache conclusion is unsupported. This script closes that hole.

Reference points:
  - MamBEV-small published (full 150-scene val): NDS 52.5 / mAP 42.2
  - StreamPETR R50-900q published (full val):    NDS 53.7 / mAP 43.2
  - StreamPETR measured by us (this 23-scene subset): NDS 55.05 / mAP 44.48

The subset runs ~1.3 points above StreamPETR's published number, so if MamBEV is
healthy we expect roughly 42-46 mAP here. Reading the outcome:

  ~0.42-0.46  -> model is healthy; stage 05's cache analysis stands.
  ~0.30-0.40  -> works but underperforms; find out why before citing stage 05.
  <0.30       -> something is broken (data format / weights / temporal stack);
                 stage 05's conclusions need re-examination.

Note the comparison to StreamPETR on this subset is apples-to-apples: the
enriched pkl (`nuscenes_infos_temporal_val.pkl`) covers the same 914 samples
across the same 23 scenes.

MECHANICS
---------
MamBEV is mmdet3d v1.x (mmengine Runner, Det3DDataSample) whereas StreamPETR is
the legacy 1.0rc6 stack, so this cannot reuse measure_stale_map.py. What it does
reuse is the partial-blob idea: nuScenes-devkit asserts that the predicted and
ground-truth sample-token sets are equal, but our val pkl covers 23 of the 150
val scenes, so `load_gt` is patched to return only the tokens we predict on.
Filtering GT (not predictions) is the semantically correct direction — we want
devkit-quality metrics computed on the subset we actually ran.

Usage:

    cd /ibex/user/khangea/multisensor/MamBEV
    conda activate /ibex/user/khangea/conda-environments/mambev_env
    PYTHONPATH=$PWD python ../stages/06_gated_bev_psm/measure_mambev_baseline.py \
        --config projects/MAMBEV/configs/MamBEV/mambev_small_t3.py \
        --ckpt   ../checkpoints/mambev/mambev_small_detector.pth \
        --work-dir ../stages/06_gated_bev_psm/dumps/mambev_baseline
"""

import argparse
import os
import pickle

from mmengine.config import Config
from mmengine.runner import Runner

# ── Config ────────────────────────────────────────────────────────────────

# Published reference numbers, for the summary at the end.
MAMBEV_SMALL_PUBLISHED = {"mAP": 0.422, "NDS": 0.525}
STREAMPETR_OURS_SUBSET = {"mAP": 0.4448, "NDS": 0.5505}

VAL_PKL = "data/nuscenes/nuscenes_infos_temporal_val.pkl"

# How many samples the 23-scene partial blob covers. The run is aborted if the
# dataloader disagrees — see `indices` in prepare_cfg() for why that can happen.
EXPECTED_SAMPLES = 914


# ── Partial-blob eval patch ───────────────────────────────────────────────

def allowed_tokens_from_pkl(pkl_path):
    """Sample tokens our val pkl actually covers (the 23-scene partial blob)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    items = data["data_list"] if "data_list" in data else data["infos"]
    toks = {i["token"] for i in items}
    scenes = {i.get("scene_token") for i in items}
    print(f"[baseline] val pkl: {len(toks)} samples across {len(scenes)} scenes")
    return toks


def install_partial_blob_eval_patch(allowed):
    """Patch nuScenes-devkit `load_gt` to serve only `allowed` sample tokens.

    Mirrors the patch measure_stale_map.py installs for StreamPETR. Without it
    NuScenesEval raises on `set(pred.sample_tokens) == set(gt.sample_tokens)`,
    because GT is loaded for all 150 val scenes while we predict on 23.
    """
    import nuscenes.eval.common.loaders as loaders_mod
    import nuscenes.eval.detection.evaluate as eval_mod

    orig_load_gt = loaders_mod.load_gt

    def patched_load_gt(nusc, eval_split, box_cls, verbose=False):
        boxes = orig_load_gt(nusc, eval_split, box_cls, verbose=verbose)
        before = len(boxes.boxes)
        boxes.boxes = {t: b for t, b in boxes.boxes.items() if t in allowed}
        print(f"[baseline] GT filtered {before} -> {len(boxes.boxes)} tokens")
        return boxes

    loaders_mod.load_gt = patched_load_gt
    if hasattr(eval_mod, "load_gt"):
        eval_mod.load_gt = patched_load_gt


# ── Reporting ─────────────────────────────────────────────────────────────

def report(metrics):
    """Print the measured numbers against the published/known reference points."""
    ap = nds = None
    for k, v in metrics.items():
        if k.endswith("mAP") or k.endswith("mean_ap"):
            ap = v
        if k.endswith("NDS") or k.endswith("nd_score"):
            nds = v

    print("\n" + "=" * 66)
    print("MamBEV baseline on our 23-scene val subset")
    print("=" * 66)
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]}")

    if ap is None:
        print("\n  (could not locate mAP in the metric dict — see raw keys above)")
        return

    print("-" * 66)
    print(f"{'':<34}{'mAP':>8}{'NDS':>8}")
    print(f"{'MamBEV-small published (full val)':<34}"
          f"{MAMBEV_SMALL_PUBLISHED['mAP']:>8.4f}"
          f"{MAMBEV_SMALL_PUBLISHED['NDS']:>8.4f}")
    print(f"{'StreamPETR, ours (this subset)':<34}"
          f"{STREAMPETR_OURS_SUBSET['mAP']:>8.4f}"
          f"{STREAMPETR_OURS_SUBSET['NDS']:>8.4f}")
    print(f"{'MamBEV, ours (this subset)':<34}{ap:>8.4f}"
          f"{(nds if nds is not None else float('nan')):>8.4f}")
    print("-" * 66)

    if ap >= 0.40:
        verdict = ("HEALTHY — consistent with the published number; stage 05's "
                   "BEV cache analysis rests on a working model.")
    elif ap >= 0.30:
        verdict = ("UNDERPERFORMING — works but well below published. Diagnose "
                   "before citing stage 05's conclusions.")
    else:
        verdict = ("BROKEN — far below published. Stage 05 analysed the BEV of a "
                   "model that is not detecting properly; its conclusions need "
                   "re-examination.")
    print(f"VERDICT: {verdict}")


# ── Main ──────────────────────────────────────────────────────────────────

def prepare_cfg(cfg):
    """Apply the overrides this measurement needs, leaving the upstream file alone.

    Three settings in `mambev_small_t3.py` are authored for MamBEV's own
    training loop and are wrong (or fatal) for a standalone baseline eval:

    1. `indices=500` on val_dataset (line ~480). A dev-time shortcut that
       truncates the dataset to its first 500 entries. We would have evaluated
       an arbitrary prefix of our 914 samples — not the 23 scenes — and, since
       the GT patch admits all 914 tokens, nuScenes-devkit would then fail its
       `set(pred) == set(gt)` assertion. Silent-wrong or crash, neither useful.

    2. `visualizer` uses `WandbVisBackend`, but wandb is not installed in
       mambev_env, so Runner.__init__ dies with an ImportError before any
       inference happens. A baseline eval needs no experiment tracking; drop
       the backends rather than adding a dependency.

    3. `img_backbone.init_cfg` points at `./ckpts/r101_dcn_fcos3d_pretrain.pth`,
       which does not exist in this clone. It is redundant here — `load_from`
       supplies every backbone weight from the detector checkpoint — so drop
       the init_cfg instead of materialising a file just to be overwritten.

    Note the EMA question was checked and needs no override: mmengine's
    `before_save_checkpoint` stores the *averaged* weights in `state_dict`
    ("so that we can directly load the averaged model weights for deployment"),
    and the swap-back branch of `after_load_checkpoint` only runs when
    `runner._resume` is set, which `test()` never does. Both paths evaluate the
    same EMA weights, so `EMAHook` is left in place.
    """
    for loader in ("test_dataloader", "val_dataloader"):
        if loader in cfg and "indices" in cfg[loader]["dataset"]:
            n = cfg[loader]["dataset"]["indices"]
            cfg[loader]["dataset"]["indices"] = None
            print(f"[baseline] {loader}: indices={n} -> None (use all samples)")

    cfg.visualizer = dict(type="Visualizer", vis_backends=[])

    backbone = cfg.model.get("img_backbone", {})
    if backbone.get("init_cfg") is not None:
        backbone["init_cfg"] = None
        print("[baseline] dropped img_backbone.init_cfg (load_from supplies it)")

    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",
                    default="projects/MAMBEV/configs/MamBEV/mambev_small_t3.py",
                    help="train config; the _inference variant references "
                         "classes that do not exist upstream (stage 05 note)")
    ap.add_argument("--ckpt",
                    default="../checkpoints/mambev/mambev_small_detector.pth")
    ap.add_argument("--work-dir",
                    default="../stages/06_gated_bev_psm/dumps/mambev_baseline")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    cfg.load_from = args.ckpt
    cfg.work_dir = args.work_dir
    cfg = prepare_cfg(cfg)
    os.makedirs(args.work_dir, exist_ok=True)

    # Import the plugin so MamBEV's detectors/metrics are registered.
    if cfg.get("custom_imports"):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    install_partial_blob_eval_patch(allowed_tokens_from_pkl(VAL_PKL))

    runner = Runner.from_cfg(cfg)

    # Fail loudly rather than reporting a number computed on the wrong samples:
    # any mismatch here means the eval is not comparable to StreamPETR's, which
    # ran on exactly these 914 tokens.
    n = len(runner.test_dataloader.dataset)
    print(f"[baseline] test dataset: {n} samples (expected {EXPECTED_SAMPLES})")
    if n != EXPECTED_SAMPLES:
        raise SystemExit(
            f"error: dataset has {n} samples, expected {EXPECTED_SAMPLES}. "
            "Check the `indices` override in prepare_cfg().")

    metrics = runner.test()
    report(metrics)


if __name__ == "__main__":
    main()
