"""Phase 3.1 sanity-check: assert dumped scenes obey the latent + feature
extraction contract documented in dump_latents.py.

Catches silent misalignment bugs early:

  * Per-frame counts match between memory snapshots and image features.
  * Memory tensor shapes are self-consistent across frames.
  * `img_feats` are fp16, in the expected shape `[6, 256, 16, 44]`, with
    no NaN/Inf, and have plausible value range (post-FPN features are
    typically ~unit-scale; sanity-check that none of the 6 cams is
    saturating).
  * `timestamp` is monotonically non-decreasing (the scene was walked in
    frame order, no reordering).
  * `schema_version` field is present and ≥ 2.
  * Report total size on disk + estimated per-scene + per-100-scene
    storage so we know what scale-up will cost.

Run:
    python stages/04_streampetr_latent/verify_schema.py \
        stages/04_streampetr_latent/dumps/scene_0.pt
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

EXPECTED_IMG_FEAT_SHAPE = (6, 256, 16, 44)
EXPECTED_M = 1280               # num_propagated + memory_len
EXPECTED_D = 256


def _check_frame(f: dict, idx: int, prev_ts: Optional[float]) -> list:
    """Returns a list of failure strings (empty if all OK)."""
    fails = []

    # --- memory tensors --------------------------------------------------
    me = f["memory_embedding"]
    if tuple(me.shape) != (EXPECTED_M, EXPECTED_D):
        fails.append(f"frame {idx}: memory_embedding shape {tuple(me.shape)} "
                     f"!= ({EXPECTED_M}, {EXPECTED_D})")
    if torch.isnan(me).any() or torch.isinf(me).any():
        fails.append(f"frame {idx}: memory_embedding has NaN/Inf")

    mr = f["memory_reference_point"]
    if tuple(mr.shape) != (EXPECTED_M, 3):
        fails.append(f"frame {idx}: memory_reference_point shape "
                     f"{tuple(mr.shape)} != ({EXPECTED_M}, 3)")

    # --- img_feats -------------------------------------------------------
    img = f.get("img_feats")
    if img is None:
        fails.append(f"frame {idx}: img_feats missing")
        return fails
    if tuple(img.shape) != EXPECTED_IMG_FEAT_SHAPE:
        fails.append(f"frame {idx}: img_feats shape {tuple(img.shape)} "
                     f"!= {EXPECTED_IMG_FEAT_SHAPE}")
    if img.dtype != torch.float16:
        fails.append(f"frame {idx}: img_feats dtype {img.dtype} != float16")
    if torch.isnan(img).any() or torch.isinf(img).any():
        fails.append(f"frame {idx}: img_feats has NaN/Inf")
    # Per-cam stats — flag suspiciously degenerate cams (all zero,
    # all-saturated, single-value).
    img_f32 = img.float()
    per_cam_std = img_f32.flatten(1).std(dim=1)
    if (per_cam_std < 1e-3).any():
        bad = [int(i) for i, s in enumerate(per_cam_std) if s < 1e-3]
        fails.append(f"frame {idx}: cams {bad} have near-zero std on img_feats")

    # --- timestamp -------------------------------------------------------
    ts = f["timestamp"]
    if prev_ts is not None and ts < prev_ts - 1e-6:
        fails.append(f"frame {idx}: timestamp went backwards "
                     f"({prev_ts:.3f} → {ts:.3f})")

    return fails


def _summarize_scene(scene_tok: str, scene: dict, dump_path: str):
    frames = scene["frames"]
    T = len(frames)

    fails = []
    if scene.get("schema_version", 1) < 2:
        fails.append(f"scene {scene_tok[:8]}: schema_version "
                     f"{scene.get('schema_version')} < 2 (no img_feats)")

    prev_ts = None
    for i, f in enumerate(frames):
        fails.extend(_check_frame(f, i, prev_ts))
        prev_ts = f["timestamp"]

    # --- size accounting ------------------------------------------------
    one_img = frames[0]["img_feats"]
    bytes_per_frame = one_img.element_size() * one_img.numel()
    bytes_per_scene_feat = bytes_per_frame * T

    on_disk = os.path.getsize(dump_path) if os.path.exists(dump_path) else 0

    print(f"\nScene {scene_tok[:8]}: {T} frames")
    print(f"  schema_version = {scene.get('schema_version')}")
    print(f"  img_feats shape per frame: {tuple(one_img.shape)} "
          f"dtype={one_img.dtype}")
    print(f"  img_feats stats (frame 0): "
          f"mean={one_img.float().mean():+.4f}  "
          f"std={one_img.float().std():.4f}  "
          f"min={one_img.float().min():+.4f}  "
          f"max={one_img.float().max():+.4f}")
    print(f"  per-frame img_feats size : {bytes_per_frame/1024:.1f} KB")
    print(f"  scene img_feats subtotal : {bytes_per_scene_feat/1024**2:.1f} MB")
    print(f"  whole .pt on disk        : {on_disk/1024**2:.1f} MB")
    print(f"  projected for 50 scenes  : "
          f"≈ {50 * bytes_per_scene_feat/1024**3:.2f} GB feat-only")

    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help=".pt produced by dump_latents.py")
    args = ap.parse_args()

    dump = torch.load(args.dump, map_location="cpu", weights_only=False)
    all_fails = []
    for scene_tok, scene in dump.items():
        all_fails.extend(_summarize_scene(scene_tok, scene, args.dump))

    if all_fails:
        print("\nFAIL — schema violations:")
        for f in all_fails:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nOK — schema contract satisfied on all scenes.")


if __name__ == "__main__":
    main()
