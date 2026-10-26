"""Phase 4 pilot — generate (stale, fresh) training tuples for LTC,
**context-matched edition**.

Why this differs from the earlier two-walk version:

The first pilot attempt walked each scene twice — once entirely fresh
(`F_t` trajectory), once entirely stale (`S_t` trajectory) — and paired
`(S_t inputs, F_t target)`. Training on that succeeded in L2 (val_emb
4.74 → 4.00) but the trained LTC HURT mAP at inference time. Diagnosis:
inference always runs in stale trajectory, so the LTC sees `S_t`
carried context but its target lives in `F_t` context. Writing a near-
`F_t[0:256]` prediction into a memory whose `[256:1280]` is `S_t`
produces a CHIMERA state. The head's cross-attention was never trained
on this. mAP collapses.

Fix: pair the target with the SAME carried context the input has. At
each stale-trajectory frame `t`:

  1. Save head memory pre-forward.
  2. Run head with STALE features → capture `S_t[0:256]` inputs.
  3. Save head memory post-stale-forward.
  4. Restore pre-forward memory.
  5. Disable substitutor; run head with FRESH features → capture
     `[0:256]` as the TARGET. Memory in this run shares slots
     `[256:1280]` with the stale trajectory.
  6. Restore post-stale memory and continue stale trajectory.

The target is "what the head WOULD have produced at frame `t` if given
fresh features but with the SAME (stale-context) memory state". This
is exactly what the LTC will be asked to predict at inference time.
No chimera.

Run:
    cd /ibex/user/khangea/multisensor/StreamPETR
    PYTHONPATH=$PWD python ../stages/04_streampetr_latent/dump_ltc_pairs.py \
        --config projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
        --ckpt   ../checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth \
        --delays 1 2 3 \
        --out    ../stages/04_streampetr_latent/dumps/ltc_pairs_pilot_v2.pt
"""

import argparse
import os
import sys
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from measure_stale_map import (  # noqa: E402
    N_CAMS,
    FeatureSubstitutor,
    _compute_scene_starts,
)


NUM_PROPAGATED = 256
MEMORY_LEN = 1024


_MEM_FIELDS = (
    "memory_embedding",
    "memory_reference_point",
    "memory_timestamp",
    "memory_egopose",
    "memory_velo",
)


def _build_model_and_data(config_path, ckpt_path):
    cfg = Config.fromfile(config_path)
    cfg.data.test.test_mode = True
    spg = cfg.data.test.pop("samples_per_gpu", 1)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    dataset = build_dataset(cfg.data.test)
    model.module.CLASSES = dataset.CLASSES
    loader = build_dataloader(
        dataset, samples_per_gpu=spg, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return cfg, model, dataset, loader


def _clone_head_memory(head, detector):
    snap = {f: (getattr(head, f).clone() if getattr(head, f) is not None else None)
            for f in _MEM_FIELDS}
    snap["prev_scene_token"] = detector.prev_scene_token
    return snap


def _restore_head_memory(head, detector, snap):
    for f in _MEM_FIELDS:
        setattr(head, f, snap[f])
    detector.prev_scene_token = snap["prev_scene_token"]


def _walk_paired(model, loader, dataset, scene_starts, delay):
    """Single walk producing context-matched (stale, fresh) pairs."""
    delays = [delay] * N_CAMS
    sub = FeatureSubstitutor(model, scene_starts, delays)
    detector = model.module
    head = detector.pts_bbox_head
    detector.prev_scene_token = None
    head.reset_memory()

    pairs = []
    scene_start_set = set(scene_starts)
    frame_in_scene = 0
    cur_scene_tok = None
    sample_idx = 0
    t0 = time.time()

    for batch in loader:
        info = dataset.data_infos[sample_idx]
        if sample_idx in scene_start_set:
            frame_in_scene = 0
            cur_scene_tok = info.get("scene_token") or info.get("scene_name")

        # Save pre-forward memory state.
        pre_state = _clone_head_memory(head, detector)

        # STALE pass (substitutor enabled, will apply staleness)
        sub.enabled = True
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **batch)
        was_stale = sub.was_stale

        stale_snap = {
            "stale_new_emb":  head.memory_embedding[0, :NUM_PROPAGATED, :].half().cpu(),
            "stale_new_ref":  head.memory_reference_point[0, :NUM_PROPAGATED, :].half().cpu(),
            "stale_new_velo": head.memory_velo[0, :NUM_PROPAGATED, :].half().cpu(),
            "carried_emb":    head.memory_embedding[0, NUM_PROPAGATED:NUM_PROPAGATED + MEMORY_LEN, :].half().cpu(),
            "carried_ref":    head.memory_reference_point[0, NUM_PROPAGATED:NUM_PROPAGATED + MEMORY_LEN, :].half().cpu(),
        }
        post_stale_state = _clone_head_memory(head, detector)

        # FRESH-with-stale-context pass: only meaningful at frames whose
        # stale pass actually substituted. Frame 0 of every scene falls
        # back to fresh due to cache-clamp, so was_stale=False and we
        # skip — no fresh-vs-stale delta to learn.
        if was_stale:
            _restore_head_memory(head, detector, pre_state)
            sub.enabled = False
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **batch)
            target_snap = {
                "fresh_new_emb":  head.memory_embedding[0, :NUM_PROPAGATED, :].half().cpu(),
                "fresh_new_ref":  head.memory_reference_point[0, :NUM_PROPAGATED, :].half().cpu(),
                "fresh_new_velo": head.memory_velo[0, :NUM_PROPAGATED, :].half().cpu(),
            }
            sub.enabled = True
            _restore_head_memory(head, detector, post_stale_state)

            pairs.append({
                "scene_token":    cur_scene_tok,
                "frame_in_scene": frame_in_scene,
                "sample_token":   info["token"],
                "k":              delay,
                **stale_snap,
                **target_snap,
            })
        frame_in_scene += 1
        sample_idx += 1

    dt = time.time() - t0
    print(f"  [delay={delay}] walked {sample_idx} samples, produced "
          f"{len(pairs)} pairs in {dt:.1f}s")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--delays", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    print(f"Configured stale delays (frames): {args.delays}")

    all_pairs = []
    for k in args.delays:
        print(f"\n== Walk for delay k={k} (context-matched) ==")
        cfg, model, dataset, loader = _build_model_and_data(args.config, args.ckpt)
        scene_starts = _compute_scene_starts(dataset)
        if k == args.delays[0]:
            print(f"Dataset: {len(dataset)} samples / {len(scene_starts)} scenes")
        pairs = _walk_paired(model, loader, dataset, scene_starts, k)
        all_pairs.extend(pairs)
        # Free model to release substitutor's monkey-patch on extract_img_feat
        del model

    print(f"\nTotal (stale, fresh) pairs: {len(all_pairs)}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"pairs": all_pairs, "delays": args.delays,
                "schema": "context-matched"}, args.out)
    print(f"Saved → {args.out}")

    size_bytes = sum(
        pair[k].element_size() * pair[k].numel()
        for pair in all_pairs
        for k in ("stale_new_emb", "stale_new_ref", "stale_new_velo",
                  "carried_emb", "carried_ref",
                  "fresh_new_emb", "fresh_new_ref", "fresh_new_velo")
    )
    print(f"Tensor storage: {size_bytes / 1024**2:.1f} MB across {len(all_pairs)} pairs")


if __name__ == "__main__":
    main()
