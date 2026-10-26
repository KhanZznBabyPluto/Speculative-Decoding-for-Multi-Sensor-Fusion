"""Phase 1.1 — dump per-frame dense BEV from a frozen MamBEV teacher.

For each val scene we touch:
  - walk frames in scene order,
  - run MamBEV forward (predict mode),
  - capture the head's BEV tensor via the same hook used in smoke_test.py,
  - record ego2global + timestamp + sample_token,
  - save one `.pt` per scene.

Schema (SCHEMA_VERSION = 1):
{
  "scene_token": str,
  "frames": [
      {
          "sample_token":   str,
          "timestamp_us":   int,                   # nuScenes microseconds
          "ego2global":     Tensor[4,4]  fp64,     # global<-ego at this frame
          "bev":            Tensor[C,H,W] fp16,    # MamBEV BEV (post-head BEV encoder)
          "bev_layout":     "chw" | "lc",          # which axis order we stored
          "bev_h":          int,
          "bev_w":          int,
          "real_h":         float,                 # metres
          "real_w":         float,                 # metres
      },
      ...
  ],
}

We store fp16 for `bev` ego2global is fp64 — the warp utility
takes the inverse and we don't want quat round-off.

Run from MamBEV/:

    cd /ibex/user/khangea/multisensor/MamBEV
    conda activate /ibex/user/khangea/conda-environments/mambev_env
    PYTHONPATH=$PWD python ../stages/06_gated_bev_psm/dump_bev.py \\
        --config <cfg> --ckpt <ckpt> --out ../stages/06_gated_bev_psm/dumps/
"""

import argparse
import os
import sys

import numpy as np
import torch

SCHEMA_VERSION = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True,
                   help="Output directory for per-scene .pt files.")
    p.add_argument("--max-scenes", type=int, default=None,
                   help="Cap on scenes processed (debugging).")
    p.add_argument("--scene-token", type=str, default=None,
                   help="If set, only dump this single scene token.")
    return p.parse_args()


def build_model_and_dataset(config_path, ckpt_path):
    """Same as smoke_test.build_model_and_dataset — duplicate intentionally
    to keep each script independently runnable. If the body diverges, refactor
    out to a shared helper at that point, not before."""
    from mmengine.config import Config
    from mmengine.registry import MODELS, DATASETS, init_default_scope
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings

    cfg = Config.fromfile(config_path)
    ci = cfg.get("custom_imports")
    if ci is not None:
        import_modules_from_strings(**ci)
    if cfg.get("default_scope"):
        init_default_scope(cfg.default_scope)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.eval().cuda()

    dl_cfg = cfg.get("test_dataloader") or cfg.get("val_dataloader")
    dataset = DATASETS.build(dl_cfg.dataset)
    return cfg, model, dataset


class BEVCapture:
    """Capture current-frame BEV by monkey-patching the head's `.forward`.

    MamBEV's `predict()` calls `pts_bbox_head.forward(...)` directly (not
    `pts_bbox_head(...)`), bypassing the `nn.Module.__call__` machinery —
    so `register_forward_hook` never fires for the current-frame call.
    We wrap `forward` itself. See smoke_test.BEVCapture for the diagnosis.
    """
    def __init__(self):
        self.bev = None

    def install(self, head):
        if hasattr(head, "_bevcapture_orig_forward"):
            head.forward = head._bevcapture_orig_forward
        head._bevcapture_orig_forward = head.forward

        def wrapped(*args, **kwargs):
            out = head._bevcapture_orig_forward(*args, **kwargs)
            if isinstance(out, dict) and "bev_embed" in out:
                self.bev = out["bev_embed"].detach()
            return out

        head.forward = wrapped

    def remove(self, head):
        if hasattr(head, "_bevcapture_orig_forward"):
            head.forward = head._bevcapture_orig_forward
            del head._bevcapture_orig_forward


def build_ego2global(info) -> torch.Tensor:
    """Construct a 4x4 ego2global matrix from a nuScenes info dict.

    MamBEV's pipeline (mmdet3d 1.x) stores ego pose as separate
    translation + rotation (quat) fields. Stage 04's dump_latents.py:57-69
    hit this and the silent-identity fallback bug — we don't repeat it.

    Field names vary across mmdet3d versions; we try in order:
      - info['ego2global']           (already a 4x4 matrix)
      - info['lidar_points']['lidar2ego'] + per-frame 'ego2global_rotation' + 'ego2global_translation'
      - info['ego2global_translation'] + info['ego2global_rotation']    (legacy)
    If none match, raise — silent-identity is unacceptable.
    """
    from pyquaternion import Quaternion

    if "ego2global" in info and np.asarray(info["ego2global"]).shape == (4, 4):
        return torch.tensor(np.asarray(info["ego2global"]), dtype=torch.float64)

    rot = info.get("ego2global_rotation")
    trans = info.get("ego2global_translation")
    if rot is not None and trans is not None:
        # nuScenes quat is (w, x, y, z).
        q = Quaternion(rot)
        R = q.rotation_matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(trans, dtype=np.float64)
        return torch.from_numpy(T)

    raise RuntimeError(
        f"Cannot reconstruct ego2global from info; keys = {list(info.keys())}"
    )


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg, model, dataset = build_model_and_dataset(args.config, args.ckpt)

    cap = BEVCapture()
    cap.install(model.pts_bbox_head)

    # Group dataset indices by scene_token. Dataset reordering for streaming
    # (as in stage 04 Phase 0.1 sign-off) is real; we still group by
    # scene_token so dumps are scene-coherent regardless of dataset order.
    by_scene = {}
    data_infos = getattr(dataset, "data_infos", None) or getattr(dataset, "data_list")
    for idx, info in enumerate(data_infos):
        st = info.get("scene_token") or info.get("token_scene")
        if st is None:
            raise RuntimeError(f"info {idx} has no scene_token; keys: {list(info.keys())}")
        by_scene.setdefault(st, []).append(idx)

    # Sort frames within each scene by timestamp.
    for st, idxs in by_scene.items():
        idxs.sort(key=lambda i: data_infos[i]["timestamp"])

    scene_tokens = list(by_scene.keys())
    if args.scene_token is not None:
        scene_tokens = [args.scene_token]
    if args.max_scenes is not None:
        scene_tokens = scene_tokens[: args.max_scenes]

    print(f"Dumping {len(scene_tokens)} scene(s) to {args.out}")

    from mmengine.dataset import default_collate

    with torch.no_grad():
        for s_i, st in enumerate(scene_tokens):
            frames_out = []
            for idx in by_scene[st]:
                info = data_infos[idx]
                sample = dataset[idx]
                batch = default_collate([sample])
                data = model.data_preprocessor(batch, training=False)
                _ = model._run_forward(data, mode="predict")

                bev = cap.bev
                if not isinstance(bev, torch.Tensor):
                    raise RuntimeError("Hook did not capture a Tensor; "
                                       "specialize BEVCapture per smoke_test output.")

                # Normalize to [C, H, W]. Layout decision is set by smoke_test.
                if bev.dim() == 3 and bev.shape[0] == 1:
                    # [1, L, C] or [1, C, H*W]? Defer to layout from cfg.
                    bev = bev.squeeze(0)
                # If [L, C] reshape to [C, H, W] using bev_h, bev_w from head.
                if bev.dim() == 2:
                    bev_h = model.pts_bbox_head.bev_h
                    bev_w = model.pts_bbox_head.bev_w
                    bev = bev.transpose(0, 1).reshape(-1, bev_h, bev_w)
                # Now expect [C, H, W].
                assert bev.dim() == 3, f"Unexpected BEV shape: {tuple(bev.shape)}"

                frames_out.append({
                    "sample_token": info.get("token") or info.get("sample_token"),
                    "timestamp_us": int(info["timestamp"]),
                    "ego2global":   build_ego2global(info),
                    "bev":          bev.detach().to(torch.float16).cpu(),
                    "bev_layout":   "chw",
                    "bev_h":        int(bev.shape[1]),
                    "bev_w":        int(bev.shape[2]),
                    # Metric extents — read from head if exposed; otherwise
                    # filled by hand from config (real_h, real_w in MamBEV).
                    "real_h":       float(getattr(model.pts_bbox_head, "real_h", 102.4)),
                    "real_w":       float(getattr(model.pts_bbox_head, "real_w", 102.4)),
                })

            out_path = os.path.join(args.out, f"scene_{s_i:03d}_{st[:8]}.pt")
            torch.save({
                "schema_version": SCHEMA_VERSION,
                "scene_token": st,
                "frames": frames_out,
            }, out_path)
            print(f"  [{s_i+1}/{len(scene_tokens)}] {st[:8]} -> "
                  f"{out_path}  ({len(frames_out)} frames)")

    cap.remove(model.pts_bbox_head)
    print("Done.")


if __name__ == "__main__":
    main()
