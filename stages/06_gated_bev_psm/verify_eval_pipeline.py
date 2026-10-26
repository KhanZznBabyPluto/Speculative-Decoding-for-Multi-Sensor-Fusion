"""Sanity check #1 — eval pipeline smoke test.

The whole stage 06 plan rests on this claim:

    For any BEV tensor M we produce, we can pipe M through MamBEV's frozen
    detection head, get a Det3DDataSample, run nuScenes-devkit evaluation,
    and get mAP/NDS numbers that we trust.

This script verifies it on M = the teacher's own BEV (i.e., the dump we just
made in dump_bev.py). If teacher-BEV-fed-back gives roughly the same mAP/NDS
as MamBEV's normal forward (within rounding noise), the eval plumbing is
healthy. If not — the eval plumbing is broken and every downstream number
will be wrong.

What we do:
  1. Run MamBEV's normal `tools/test.py` path on the same scenes we dumped
     → get "reference mAP" (the teacher's natural number).
  2. Run a parallel path where we *replace* the head's BEV-encoder output
     with the dumped teacher BEV, then run the remainder of the head
     (decoder, classification, regression) and the same nuScenes-devkit
     evaluator → get "round-trip mAP".
  3. Assert |round_trip - reference| < ε on the same sample set.

Run from MamBEV/:

    cd /ibex/user/khangea/multisensor/MamBEV
    conda activate /ibex/user/khangea/conda-environments/mambev_env
    PYTHONPATH=$PWD python ../stages/06_gated_bev_psm/verify_eval_pipeline.py \\
        --config <cfg> --ckpt <ckpt> \\
        --dumps  ../stages/06_gated_bev_psm/dumps/ \\
        --max-scenes 5
"""

import argparse
import glob
import os
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dumps", required=True,
                   help="Directory of per-scene .pt files from dump_bev.py.")
    p.add_argument("--max-scenes", type=int, default=5,
                   help="Run on a small subset first; expand once it passes.")
    p.add_argument("--epsilon", type=float, default=1e-3,
                   help="Max acceptable |round_trip_mAP - reference_mAP|.")
    return p.parse_args()


def load_dumps(dumps_dir: str, max_scenes: int):
    """Load (scene_token -> {sample_token -> bev_tensor}) from dump_bev.py output."""
    paths = sorted(glob.glob(os.path.join(dumps_dir, "scene_*.pt")))[:max_scenes]
    table = {}
    for p in paths:
        rec = torch.load(p, map_location="cpu")
        per_token = {f["sample_token"]: f["bev"] for f in rec["frames"]}
        table[rec["scene_token"]] = per_token
    return table


def install_bev_substitutor(model, dump_table):
    """Monkey-patch the head's BEV-encoder so it returns the dumped BEV
    instead of re-computing it from camera features.

    Concretely, the head computes BEV at the start of its `forward(...)` via
    something like
        bev = self.transformer.get_bev_features(mlvl_feats, bev_queries, ...)
    We wrap whatever returns the BEV with a function that, given the current
    sample_token (read from img_metas), looks up the dumped BEV and returns it.

    Exact target depends on MamBEV's head — pin down during smoke_test, then
    fill this in. Until then we leave the hook stub and raise on call.
    """
    raise NotImplementedError(
        "Wire this once smoke_test.py confirms the BEV path. "
        "Likely targets: model.pts_bbox_head.transformer.get_bev_features, "
        "or the bev_encoder submodule. Substitute return value with "
        "dump_table[scene_token][sample_token] reshaped to the head's BEV layout."
    )


def run_reference_eval(model, dataset, cfg):
    """Run MamBEV's normal forward through nuScenes-devkit evaluation.
    Returns (overall_mAP, overall_NDS, per_token_dict)."""
    # Use Runner.test() or the dataset's evaluator directly. MamBEV uses
    # mmengine evaluator — see tools/test.py for the canonical setup.
    raise NotImplementedError(
        "Mirror MamBEV/tools/test.py here, but restrict to the dumped scenes. "
        "Hook into the head normally — no substitution."
    )


def run_round_trip_eval(model, dataset, cfg, dump_table):
    """Run MamBEV with the BEV-encoder substituted to return dumped BEV.
    Returns (overall_mAP, overall_NDS, per_token_dict)."""
    raise NotImplementedError(
        "Same as run_reference_eval, but install_bev_substitutor first."
    )


def main():
    args = parse_args()
    dumps = load_dumps(args.dumps, args.max_scenes)
    print(f"Loaded {len(dumps)} scenes from {args.dumps}")

    # The two paths are stubs until smoke_test pins down the BEV entry point
    # — we expose the API now so the wiring is unambiguous later.
    print(
        "\nThis script is a stub. Fill `install_bev_substitutor`, "
        "`run_reference_eval`, `run_round_trip_eval` after smoke_test.py "
        "exposes MamBEV's BEV-producing call site.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
