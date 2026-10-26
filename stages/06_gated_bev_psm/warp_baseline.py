"""Sanity check #2 — pure-warp baseline.

Stage 06 v1 (the warp+smoother test) only makes sense if a *learned*
smoother can beat the trivial no-learning warp. This script measures
the no-learning ceiling.

Two metrics, both on the dumped teacher BEV (no model training involved):

  (A) BEV-space MSE floor.
      For each consecutive frame pair (t-1, t) in each dumped scene:
          M_warped = warp_bev(M_{t-1}, ego_{t-1}, ego_t)
          mse[t] = MSE(M_warped, M_t)
      Report:
          mean MSE over all pairs,
          MSE distribution stratified by ego-motion magnitude
              (how far did the vehicle move between t-1 and t).
      This is the *MSE floor* for the ConvGRU at single-step (N=1) horizon —
      any learnable smoother must beat this.

  (B) Detection-head mAP floor under purely-warped state for N ∈ {1,2,4,8}.
      For each scene, simulate "skip backbone for N-1 of every N frames":
          frame 0:           M = teacher M_0     (full backbone)
          frames 1..N-1:     M ← warp_bev(M, ego)        (no backbone)
          frame N:           M = teacher M_N     (refresh)
          frames N+1..2N-1:  M ← warp_bev(M, ego)
          ...
      Pipe each (warped or refreshed) M through the frozen detection head,
      collect detections, run nuScenes evaluation, get mAP/NDS per N.

(A) is offline and fast (just torch). (B) requires the same head-substitution
plumbing as verify_eval_pipeline.py — if (B) is implemented but
verify_eval_pipeline.py isn't passing, the (B) numbers cannot be trusted.

The two numbers together set the v1 success criterion:

    "Learned warp+smoother is interesting iff it beats pure warp on (B) at
     N=4 by ≥ 0.02 mAP. Otherwise we don't have a story even before
     touching the sensor-input question."

Sub-result of interest: how big is the gap between N=1 (=mAP @ teacher BEV
fed through frozen head) and pure-warp at N=2 / N=4 / N=8? If pure warp at
N=4 is already within 90% of teacher-mAP, v1 is uninteresting and we should
skip to v2 (sensor input) directly.

Run from MamBEV/:

    cd /ibex/user/khangea/multisensor/MamBEV
    conda activate /ibex/user/khangea/conda-environments/mambev_env
    PYTHONPATH=$PWD python ../stages/06_gated_bev_psm/warp_baseline.py \\
        --dumps ../stages/06_gated_bev_psm/dumps/ \\
        --config <cfg> --ckpt <ckpt> \\
        --refresh-every 1 2 4 8

The (A) metric can be computed without --config/--ckpt (pure offline) by
running with --mse-only.
"""

import argparse
import glob
import os
import sys

import torch

from bev_warp import BEVGrid, warp_bev


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", required=True,
                   help="Directory of per-scene .pt files from dump_bev.py.")
    p.add_argument("--mse-only", action="store_true",
                   help="Skip (B); just compute the MSE floor.")
    p.add_argument("--config", default=None,
                   help="MamBEV config. Required if not --mse-only.")
    p.add_argument("--ckpt", default=None,
                   help="MamBEV checkpoint. Required if not --mse-only.")
    p.add_argument("--refresh-every", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="N values to sweep for the (B) experiment.")
    p.add_argument("--max-scenes", type=int, default=None,
                   help="Cap scenes processed for (B); (A) always uses all.")
    return p.parse_args()


def load_scene(path):
    return torch.load(path, map_location="cpu")


def make_grid_from_frame(frame) -> BEVGrid:
    """One BEVGrid per scene is enough — all frames share the same metadata."""
    return BEVGrid(
        h=int(frame["bev_h"]),
        w=int(frame["bev_w"]),
        real_h=float(frame["real_h"]),
        real_w=float(frame["real_w"]),
    )


def measure_mse_floor(dumps_dir: str):
    """(A) BEV-space MSE floor across all dumped scene pairs."""
    paths = sorted(glob.glob(os.path.join(dumps_dir, "scene_*.pt")))
    all_mse = []
    all_disp = []
    n_pairs = 0
    for p in paths:
        rec = load_scene(p)
        frames = rec["frames"]
        if len(frames) < 2:
            continue
        grid = make_grid_from_frame(frames[0])
        for i in range(1, len(frames)):
            f_prev, f_curr = frames[i - 1], frames[i]
            bev_prev = f_prev["bev"].float().unsqueeze(0)               # [1, C, H, W]
            bev_curr = f_curr["bev"].float().unsqueeze(0)
            e_prev = f_prev["ego2global"].unsqueeze(0).float()           # [1, 4, 4]
            e_curr = f_curr["ego2global"].unsqueeze(0).float()
            with torch.no_grad():
                bev_warped = warp_bev(bev_prev, e_prev, e_curr, grid)
            mse = torch.mean((bev_warped - bev_curr) ** 2).item()
            disp = torch.linalg.norm(e_curr[0, :2, 3] - e_prev[0, :2, 3]).item()
            all_mse.append(mse)
            all_disp.append(disp)
            n_pairs += 1
    if not n_pairs:
        print("No consecutive frame pairs found; no (A) result.")
        return None
    import numpy as np
    mse_arr = np.array(all_mse)
    disp_arr = np.array(all_disp)
    print(f"\n(A) BEV-space MSE floor (pure warp, no learning):")
    print(f"    n_pairs = {n_pairs}")
    print(f"    MSE  mean = {mse_arr.mean():.5f}")
    print(f"    MSE  median = {float(np.median(mse_arr)):.5f}")
    print(f"    disp mean = {disp_arr.mean():.3f} m,  "
          f"max = {disp_arr.max():.3f} m")
    # Stratify by ego displacement buckets to expose motion-dependence.
    for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 100.0)]:
        mask = (disp_arr >= lo) & (disp_arr < hi)
        if mask.any():
            print(f"    disp [{lo:.1f}, {hi:.1f}) m: n={int(mask.sum()):4d}, "
                  f"MSE mean = {mse_arr[mask].mean():.5f}")
    return {"mse_mean": float(mse_arr.mean()), "n_pairs": n_pairs}


def measure_warp_map_floor(args):
    """(B) mAP/NDS floor under purely-warped state with periodic teacher
    refresh. Requires the head-substitution plumbing from verify_eval_pipeline.py
    — this function is intentionally stubbed until that plumbing is real."""
    raise NotImplementedError(
        "Wire after verify_eval_pipeline.py exposes the BEV-substitution "
        "entrypoint. Steps:\n"
        "  1. Group dataset samples by scene_token, sorted by timestamp.\n"
        "  2. For each scene, walk frames: at every Nth frame use teacher BEV "
        "     from the dump; in between, apply warp_bev to the running M.\n"
        "  3. Substitute M into the head via the same hook verify_eval_pipeline "
        "     uses; let the head's decoder/regressor produce detections.\n"
        "  4. Collect across scenes, run nuScenes-devkit evaluation.\n"
        "  5. Repeat for each N in args.refresh_every."
    )


def main():
    args = parse_args()
    measure_mse_floor(args.dumps)
    if args.mse_only:
        return
    if not args.config or not args.ckpt:
        print("--config and --ckpt are required for (B); pass --mse-only to "
              "skip the mAP measurement.")
        sys.exit(2)
    measure_warp_map_floor(args)


if __name__ == "__main__":
    main()
