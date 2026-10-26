"""Phase 1.1 + 3.1: dump StreamPETR's persistent latent state per frame.

For one scene at a time, run the model in test mode and, after each
frame's `simple_test` call, snapshot the head's memory state plus the
camera feature grid the head consumed. Saves as a torch .pt per scene:

    {
        "schema_version": 2,                          # bumped for img_feats
        "sample_tokens": [...],                       # in frame order
        "frames": [
            {
                "sample_token": str,
                "timestamp": float,
                "ego_pose": [4, 4] np.array (ego→global),
                "memory_embedding":       [M, D],     # B=1 squeezed
                "memory_reference_point": [M, 3],     # in GLOBAL frame
                "memory_timestamp":       [M, 1],
                "memory_egopose":         [M, 4, 4],
                "memory_velo":            [M, 2],
                "img_feats":              [N_cams, C, H, W]  fp16
            },
            ...
        ],
    }

## Feature extraction contract (Phase 3.1) — DO NOT silently change

The `img_feats` tensor MUST be captured at a fixed, documented point in
StreamPETR's pipeline. The contract:

  * Hook target  : `Petr3D.extract_img_feat` (return value).
  * Pipeline pos : POST-backbone (ResNet50), POST-neck (CPFPN). The
                   tensor is FPN-merged.
  * Level        : `position_level=0` — the head's own choice (config
                   default; not overridden). This is FPN level 0 at
                   stride 16, derived from a top-down combination of
                   ResNet C4 (1024ch, 1/16 res) and C5 (2048ch, 1/32
                   res). Confirmed via config inspection at
                   stream_petr_r50_flash_704_bs2_seq_90e.py:
                       backbone out_indices=(2, 3)
                       neck CPFPN num_outs=2, out_channels=256
                       Petr3D.position_level default 0
  * Shape        : [B=1, N_cams=6, C=256, H=16, W=44]  for 256×704
                   input (img stride 16). Stored as fp16, B squeezed →
                   [6, 256, 16, 44].
  * Temporal alignment INVARIANT: img_feats[t], memory_*[t], timestamp[t],
                   ego_pose[t] all come from THE SAME `model.forward`
                   call. There is no inter-frame averaging or smoothing
                   on the dump side.

Rationale (Phase 3 scoping discussion):
  * Single-scale, post-FPN, head-aligned. Speculator sees what the head
    sees, modulo the additional channel projections inside the head.
  * Not pre-FPN (2048ch) because that's "raw camera signal" and forces
    the speculator to re-learn perception. We are NOT learning a feature
    extractor — only a latency-correction module on the memory side.
  * Not multi-scale because that would force the speculator to learn
    multi-resolution fusion, drifting away from the actual research
    question ("how latency distorts latent scene state").

## Why post-call memory snapshot

StreamPETR's pre_update brings memory from global into ego_t coords,
then the head runs, then post_update sends memory back to global. We
capture the post-update state (in global) because:
  * it is what the next frame's pre_update will read,
  * it is invariant to ego pose between calls.

Run AFTER env_setup.sh + install_mmdet3d.sh + download_checkpoint.sh +
prepare_data.sh:

    cd /ibex/user/khangea/multisensor/StreamPETR
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    PYTHONPATH=$PWD python ../stages/04_streampetr_latent/dump_latents.py \
        --config projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
        --ckpt   ../checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth \
        --out    ../stages/04_streampetr_latent/dumps/scene_0.pt \
        --n-scenes 1
"""

SCHEMA_VERSION = 2

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from pyquaternion import Quaternion


def _ego_pose_from_info(info: dict) -> np.ndarray:
    """Build the 4x4 ego→global transform from the val pkl's split fields.

    StreamPETR's converter saves rotation as a quaternion (w, x, y, z) and
    translation as (x, y, z), NOT as a 4x4 matrix. Calling .get('ego2global')
    silently returns identity, which makes downstream ego-warp tests fail.
    """
    R = Quaternion(info["ego2global_rotation"]).rotation_matrix
    t = np.asarray(info["ego2global_translation"], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

# Register StreamPETR's plugin classes (triggers @register_module decorators):
import projects.mmdet3d_plugin  # noqa: F401


def disable_flash_attn(cfg):
    """Optional fallback for environments without flash-attn installed.
    Swaps PETRMultiheadFlashAttention -> PETRMultiheadAttention in the
    decoder's attn_cfgs. Both share state_dict keys; the swap is loss-less.

    Not used by default — flash-attn is installed in our env. Kept as a
    documented escape hatch.
    """
    attn_cfgs = (cfg.model.pts_bbox_head.transformer.decoder
                 .transformerlayers.attn_cfgs)
    cfgs = attn_cfgs if isinstance(attn_cfgs, (list, tuple)) else [attn_cfgs]
    for c in cfgs:
        if c.get("type") == "PETRMultiheadFlashAttention":
            c["type"] = "PETRMultiheadAttention"
    return cfg


@torch.no_grad()
def snapshot_memory(head, sample_token, timestamp, ego_pose, img_feats):
    """Pull the head's memory tensors to CPU. B=1 assumed.

    img_feats: tensor of shape [B=1, N_cams, C, H, W] captured by the
    extract_img_feat hook (see _install_feature_hook). Stored fp16,
    B-dim squeezed.
    """
    def squeeze(t):
        return None if t is None else t.detach().squeeze(0).cpu()
    return {
        "sample_token": sample_token,
        "timestamp": float(timestamp),
        "ego_pose": np.asarray(ego_pose, dtype=np.float64),
        "memory_embedding":       squeeze(head.memory_embedding),
        "memory_reference_point": squeeze(head.memory_reference_point),
        "memory_timestamp":       squeeze(head.memory_timestamp),
        "memory_egopose":         squeeze(head.memory_egopose),
        "memory_velo":            squeeze(head.memory_velo),
        "img_feats":              img_feats.detach().squeeze(0).half().cpu(),
    }


def _install_feature_hook(model):
    """Monkey-patch `extract_img_feat` to stash its return value.

    We capture the EXACT tensor the head consumes — same level
    (position_level), same shape, same dtype as inside the model's
    forward. The patched method is idempotent: it does the original
    computation, stashes the result on `model.module`, and returns it.
    """
    detector = model.module
    orig = detector.extract_img_feat

    def hooked(*args, **kwargs):
        out = orig(*args, **kwargs)
        detector._dump_last_img_feats = out
        return out

    detector.extract_img_feat = hooked
    detector._dump_last_img_feats = None
    return detector  # convenience handle


def group_by_scene(dataset):
    """Bucket dataset sample indices by scene_token, preserving frame order.

    Returns dict[scene_token] -> list of dataset indices.
    """
    by_scene = defaultdict(list)
    for i, info in enumerate(dataset.data_infos):
        scene_tok = info.get("scene_token") or info.get("scene_name")
        by_scene[scene_tok].append(i)
    # The dataset is usually already in order; do not re-sort.
    return dict(by_scene)


def run_scene(model, dataset, indices, samples_per_gpu=1):
    """Iterate a single scene's frames in order and snapshot per-frame state.

    Uses the same collate pipeline as tools/test.py / single_gpu_test, NOT
    manual DataContainer unwrapping. That was the source of subtle bugs in
    earlier versions.
    """
    from torch.utils.data import Subset, DataLoader
    from functools import partial
    from mmcv.parallel import collate

    head = model.module.pts_bbox_head
    detector = _install_feature_hook(model)
    detector.prev_scene_token = None      # force reset on first frame
    head.reset_memory()

    sub = Subset(dataset, indices)
    loader = DataLoader(
        sub,
        batch_size=samples_per_gpu,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
    )

    snapshots = []
    for i, batch in enumerate(loader):
        idx = indices[i]
        info = dataset.data_infos[idx]
        sample_token = info["token"]
        timestamp = (info["timestamp"] / 1e6
                     if info["timestamp"] > 1e10 else info["timestamp"])
        ego_pose = _ego_pose_from_info(info)

        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **batch)

        img_feats = detector._dump_last_img_feats
        assert img_feats is not None, (
            "feature hook did not capture img_feats — did extract_img_feat run?")
        snapshots.append(
            snapshot_memory(head, sample_token, timestamp, ego_pose, img_feats))

    return snapshots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--n-scenes", type=int, default=1,
                    help="how many scenes to dump (val set, in order)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-flash", action="store_true",
                    help="patch config to disable flash-attn before build")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if args.no_flash:
        cfg = disable_flash_attn(cfg)
    # Match tools/test.py: must set test_mode and pop samples_per_gpu BEFORE
    # building the dataset, or the dataset comes up in train mode.
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)

    # Build model in test mode
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Build val dataset (full pipeline)
    dataset = build_dataset(cfg.data.test)
    print(f"Loaded val dataset with {len(dataset)} samples")

    by_scene = group_by_scene(dataset)
    scene_tokens = list(by_scene.keys())[: args.n_scenes]
    print(f"Dumping {len(scene_tokens)} scenes")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_dumps = {}
    for s_tok in scene_tokens:
        idxs = by_scene[s_tok]
        print(f"  scene {s_tok[:8]}: {len(idxs)} frames")
        snaps = run_scene(model, dataset, idxs, samples_per_gpu=samples_per_gpu)
        all_dumps[s_tok] = {
            "schema_version": SCHEMA_VERSION,
            "sample_tokens": [s["sample_token"] for s in snaps],
            "frames": snaps,
        }

    torch.save(all_dumps, args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
