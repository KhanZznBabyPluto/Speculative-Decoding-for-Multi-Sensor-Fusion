"""Stage 05 — Scalable AI deliverable, per-scene runner.

For one nuScenes scene, walks all valid anchor positions and for each
k in --k-list runs MamBEV.predict() at frame t and frame t+k, captures
both BEV encoder outputs via forward hook, computes per-pair statistics
(cosine, L2, ego transform decomposition, top-1 detection score, BEV
magnitudes), and appends one JSONL line per measurement to --out.

Designed to be one slot of a SLURM array job: keep the model loaded
once per process, process all anchors in the scene, write small output
(NO tensor dumps).

Usage:
  python run_bev_stats.py \
      --scene 41fde20fedcd4d22ab26811688612870 \
      --anchor-stride 1 \
      --k-list 1 \
      --out /ibex/.../dumps/scale/scene_41fde20f.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT   = Path("/ibex/user/khangea/multisensor")
MAMBEV_ROOT = REPO_ROOT / "MamBEV"
CKPT_DIR    = REPO_ROOT / "checkpoints" / "mambev"
NUSC_ROOT   = REPO_ROOT / "data_full"

CONFIG = MAMBEV_ROOT / "projects/MAMBEV/configs/MamBEV/mambev_small_t3.py"
CKPT   = CKPT_DIR / "mambev_small_detector.pth"

# Test pipeline output dims (see dump_bev_pair.py docstring)
SRC_H, SRC_W = 900, 1600
SCALE        = 0.8
TGT_H, TGT_W = 736, 1280
N_CAMS       = 6
BEV_H, BEV_W = 100, 100
FRAMES       = (-2, -1, 0)
CAM_NAMES    = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK",  "CAM_BACK_LEFT",   "CAM_BACK_RIGHT",
)
IMG_MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)
IMG_STD  = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def setup_path():
    sys.path.insert(0, str(MAMBEV_ROOT))


# --------------------------------------------------------------------------- #
#  nuScenes helpers (lifted from dump_bev_pair.py)                            #
# --------------------------------------------------------------------------- #

def load_nuscenes():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)


def enumerate_scene_samples(nusc, scene_token: str) -> List[dict]:
    scene = nusc.get("scene", scene_token)
    cur = nusc.get("sample", scene["first_sample_token"])
    chain = [cur]
    while cur["next"]:
        cur = nusc.get("sample", cur["next"])
        chain.append(cur)
    return chain


def sample_has_all_cams_on_disk(nusc, sample: dict) -> bool:
    for cam in CAM_NAMES:
        sd = nusc.get("sample_data", sample["data"][cam])
        if not (NUSC_ROOT / sd["filename"]).exists():
            return False
    return True


def quat_T_to_matrix(translation, rotation_wxyz) -> np.ndarray:
    from pyquaternion import Quaternion
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    M[:3, 3]  = np.asarray(translation, dtype=np.float64)
    return M


def compute_lidar2img(nusc, sample: dict) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_eg = nusc.get("ego_pose",          lidar_sd["ego_pose_token"])
    lidar2ego        = quat_T_to_matrix(lidar_cs["translation"],  lidar_cs["rotation"])
    ego2global_lidar = quat_T_to_matrix(lidar_eg["translation"],  lidar_eg["rotation"])

    lidar2img_per_cam = []
    for cam in CAM_NAMES:
        cam_sd = nusc.get("sample_data", sample["data"][cam])
        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_eg = nusc.get("ego_pose",          cam_sd["ego_pose_token"])
        cam2ego_cam    = quat_T_to_matrix(cam_cs["translation"], cam_cs["rotation"])
        ego2global_cam = quat_T_to_matrix(cam_eg["translation"], cam_eg["rotation"])
        ego2cam        = np.linalg.inv(cam2ego_cam)
        global2ego_cam = np.linalg.inv(ego2global_cam)
        lidar2cam      = ego2cam @ global2ego_cam @ ego2global_lidar @ lidar2ego
        K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
        K_resized = K.copy()
        K_resized[0, :] *= SCALE
        K_resized[1, :] *= SCALE
        K_pad = np.eye(4, dtype=np.float64)
        K_pad[:3, :3] = K_resized
        lidar2img_per_cam.append((K_pad @ lidar2cam).astype(np.float32))
    return lidar2img_per_cam, ego2global_lidar.astype(np.float32), lidar2ego.astype(np.float32)


def load_imgs(nusc, sample: dict) -> np.ndarray:
    import cv2
    out = np.zeros((N_CAMS, 3, TGT_H, TGT_W), dtype=np.float32)
    for i, cam in enumerate(CAM_NAMES):
        sd = nusc.get("sample_data", sample["data"][cam])
        path = str(NUSC_ROOT / sd["filename"])
        img = cv2.imread(path)
        new_w, new_h = int(SRC_W * SCALE), int(SRC_H * SCALE)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = (img - IMG_MEAN[None, None, :]) / IMG_STD[None, None, :]
        pad_h = TGT_H - new_h
        img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)
        out[i] = img.transpose(2, 0, 1)
    return out


# --------------------------------------------------------------------------- #
#  Data sample + hook                                                         #
# --------------------------------------------------------------------------- #

def build_data_sample(sample_token, frame_idx, scene_token,
                      lidar2img_list, ego2global, lidar2ego):
    from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, Box3DMode
    eye4 = np.eye(4, dtype=np.float32)
    meta = dict(
        sample_idx=sample_token, frame_idx=frame_idx, scene_token=scene_token,
        ego2global=ego2global, lidar2ego=lidar2ego,
        lidar2img=lidar2img_list,
        lidar2cam=[eye4.copy() for _ in range(N_CAMS)],
        cam2img  =[eye4.copy() for _ in range(N_CAMS)],
        img_shape=[(TGT_H, TGT_W, 3)] * N_CAMS,
        pad_shape=[(TGT_H, TGT_W, 3)] * N_CAMS,
        ori_shape=[(SRC_H, SRC_W, 3)] * N_CAMS,
        scale_factor=SCALE,
        img_norm_cfg=dict(mean=IMG_MEAN.tolist(), std=IMG_STD.tolist(), to_rgb=False),
        box_mode_3d=Box3DMode.LIDAR, box_type_3d=LiDARInstance3DBoxes,
        can_bus=np.zeros(18, dtype=np.float32),
        prev_bev_exists=False, timestamp=float(frame_idx),
        mono_input_dict=None, mono_ann_idx=[],
        aug_param=dict(CropResizeFlipImage_param=(0, 0, TGT_W, TGT_H, False)),
        crop_offset=(0, 0), img_crop_offset=(0, 0),
        filename=[f"{c}.jpg" for c in CAM_NAMES],
        img_path=[f"{c}.jpg" for c in CAM_NAMES],
        resize_img_shape=[(TGT_H, TGT_W, 3)] * N_CAMS,
        num_pts_feats=4, pts_filename=sample_token + ".bin",
        axis_align_matrix=eye4.copy(), mask=False,
    )
    from mmdet3d.structures import Det3DDataSample
    ds = Det3DDataSample()
    ds.set_metainfo(meta)
    return ds


# --------------------------------------------------------------------------- #
#  Stats                                                                      #
# --------------------------------------------------------------------------- #

def bev_pair_stats(bev_a, bev_b) -> Dict[str, float]:
    """`bev_a`, `bev_b` shape (groups, H*W, C). Reshape to (G, C, H, W),
    compute per-cell L2 + cosine and reduce."""
    import torch
    def to_BCHW(x):
        g, n, c = x.shape
        return x.view(g, BEV_H, BEV_W, c).permute(0, 3, 1, 2).contiguous().float()
    A, B = to_BCHW(bev_a), to_BCHW(bev_b)
    diff_norm = torch.sqrt(((A - B) ** 2).sum(dim=1) + 1e-12)
    eps = 1e-6
    num = (A * B).sum(dim=1)
    da  = torch.sqrt((A * A).sum(dim=1) + eps)
    db  = torch.sqrt((B * B).sum(dim=1) + eps)
    cos = num / (da * db)
    return dict(
        l2_mean   = float(diff_norm.mean()),
        l2_median = float(diff_norm.median()),
        l2_max    = float(diff_norm.max()),
        cos_mean  = float(cos.mean()),
        cos_median= float(cos.median()),
        frac_cos_gt_85 = float((cos > 0.85).float().mean()),
        std_a     = float(A.std()),
        std_b     = float(B.std()),
    )


def ego_delta(eg_t: np.ndarray, l2e_t: np.ndarray,
              eg_tk: np.ndarray, l2e_tk: np.ndarray) -> Dict[str, float]:
    l2g_t  = eg_t  @ l2e_t
    l2g_tk = eg_tk @ l2e_tk
    T = np.linalg.inv(l2g_t) @ l2g_tk   # lidar_t ← lidar_t+k
    dx, dy, dz = T[:3, 3]
    yaw_deg = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    return dict(dx=float(dx), dy=float(dy), dz=float(dz), dyaw=yaw_deg,
                trans=float(np.linalg.norm(T[:3, 3])))


# --------------------------------------------------------------------------- #
#  Per-anchor forward                                                         #
# --------------------------------------------------------------------------- #

def run_one(model, bev_encoder, frame_assets: List[dict], anchor_idx: int,
            scene_token: str, device) -> Tuple[any, dict, dict]:
    """Run model.predict anchored at `anchor_idx` (which must be ≥ 2),
    capture BEV via hook, return (bev_tensor_cpu, top1_info, frame_assets[anchor]_excerpt)."""
    import torch
    prev_idx = [anchor_idx - 2, anchor_idx - 1]
    batch_inputs_dict, batch_data_samples = {}, {}
    for f, src_idx in zip(FRAMES, [prev_idx[0], prev_idx[1], anchor_idx]):
        pf = frame_assets[src_idx]
        t_im = torch.from_numpy(pf["imgs"]).unsqueeze(0).to(device)
        batch_inputs_dict[f]  = dict(img=t_im)
        batch_data_samples[f] = [build_data_sample(
            sample_token=pf["sample_token"], frame_idx=f, scene_token=scene_token,
            lidar2img_list=pf["lidar2img"], ego2global=pf["ego2global"],
            lidar2ego=pf["lidar2ego"],
        )]

    captured = {}
    def hook(module, inputs, output):
        t = output if hasattr(output, "shape") else output[0]
        captured["bev"] = t.detach().to(torch.float16).cpu().clone()
    h = bev_encoder.register_forward_hook(hook)
    try:
        with torch.no_grad():
            results = model.predict(batch_inputs_dict, batch_data_samples)
            torch.cuda.synchronize()
    finally:
        h.remove()

    sample = results[0]
    pi = sample.pred_instances_3d
    scores = pi.scores_3d.detach().cpu()
    labels = pi.labels_3d.detach().cpu()
    topk = torch.topk(scores, k=5)
    top1_info = dict(
        top1_score=float(topk.values[0]),
        top1_label=int(labels[topk.indices[0]]),
        top5_scores=[float(s) for s in topk.values],
    )
    return captured["bev"], top1_info, frame_assets[anchor_idx]


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="nuScenes scene token")
    parser.add_argument("--anchor-stride", type=int, default=1,
                        help="step between anchor positions (1 = every valid anchor)")
    parser.add_argument("--k-list", type=int, nargs="+", default=[1],
                        help="frame gaps to evaluate (e.g., --k-list 1 2 3)")
    parser.add_argument("--max-anchors", type=int, default=None,
                        help="cap number of anchors (useful for strong-scaling probe)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output JSONL path")
    args = parser.parse_args()

    setup_path()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[scene={args.scene}] starting; anchor_stride={args.anchor_stride}, "
          f"k_list={args.k_list}, out={args.out}", flush=True)

    # ---- nuScenes scan ----
    nusc = load_nuscenes()
    chain = enumerate_scene_samples(nusc, args.scene)
    on_disk = [sample_has_all_cams_on_disk(nusc, s) for s in chain]
    n_chain = len(chain)
    print(f"[scene={args.scene}] {n_chain} samples in scene; "
          f"{sum(on_disk)} have all 6 cams on disk", flush=True)

    # Per-frame asset cache
    max_k = max(args.k_list)
    needed_window = 2 + max_k + 1  # frames -2..0 prev + anchor + k future
    # find anchor positions where (anchor-2..anchor+k) are all valid
    valid_anchors = []
    for anchor in range(2, n_chain - max_k):
        if all(on_disk[anchor - 2 + j] for j in range(needed_window)):
            valid_anchors.append(anchor)
    valid_anchors = valid_anchors[::args.anchor_stride]
    if args.max_anchors:
        valid_anchors = valid_anchors[:args.max_anchors]
    if not valid_anchors:
        print(f"[scene={args.scene}] no valid anchor positions; exiting.", flush=True)
        return 0
    print(f"[scene={args.scene}] valid anchors: {len(valid_anchors)} "
          f"(positions {valid_anchors[0]}..{valid_anchors[-1]})", flush=True)

    # Cache assets for unique sample indices we'll touch
    unique_idx = set()
    for anchor in valid_anchors:
        for j in range(-2, max_k + 1):
            unique_idx.add(anchor + j)
    unique_idx = sorted(unique_idx)
    print(f"[scene={args.scene}] caching {len(unique_idx)} sample assets...", flush=True)
    frame_assets: Dict[int, dict] = {}
    for idx in unique_idx:
        samp = chain[idx]
        imgs = load_imgs(nusc, samp)
        lidar2img, ego2global, lidar2ego = compute_lidar2img(nusc, samp)
        frame_assets[idx] = dict(
            sample_token=samp["token"], imgs=imgs,
            lidar2img=lidar2img, ego2global=ego2global, lidar2ego=lidar2ego,
        )
    print(f"[scene={args.scene}] assets cached in "
          f"{time.time() - t_start:.1f}s", flush=True)

    # ---- Fast-fail if the GPU's compute capability isn't in our compile target
    import torch as _torch
    if not _torch.cuda.is_available():
        print(f"[scene={args.scene}] FAIL: no CUDA device visible", flush=True)
        return 1
    cap = _torch.cuda.get_device_capability(0)
    if cap < (8, 0):
        name = _torch.cuda.get_device_name(0)
        print(f"[scene={args.scene}] FAIL: GPU {name} compute capability {cap} "
              f"is below our compile target (sm_80+). mmcv/mamba-ssm kernels "
              f"will crash. Re-submit with --gres=gpu:a100:1 or recompile with "
              f"TORCH_CUDA_ARCH_LIST including this arch.", flush=True)
        return 1

    # ---- Model ----
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.MAMBEV.mambev  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import MODELS
    from mmengine.runner import load_checkpoint
    import torch

    cfg   = Config.fromfile(str(CONFIG))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(CKPT), map_location="cpu", strict=False)
    device = torch.device("cuda")
    model = model.to(device).eval()
    bev_encoder = model.pts_bbox_head.transformer.encoder
    print(f"[scene={args.scene}] model loaded "
          f"({time.time() - t_start:.1f}s elapsed)", flush=True)

    # ---- Anchor loop ----
    bev_cache: Dict[int, "torch.Tensor"] = {}
    top_cache: Dict[int, dict]           = {}

    def get_bev(anchor: int):
        if anchor in bev_cache:
            return bev_cache[anchor], top_cache[anchor]
        bev, top, _ = run_one(model, bev_encoder, frame_assets, anchor,
                              args.scene, device)
        bev_cache[anchor] = bev
        top_cache[anchor] = top
        return bev, top

    n_written = 0
    with open(args.out, "w") as fh:
        for ai, anchor in enumerate(valid_anchors):
            t_anchor = time.time()
            bev_t, top_t = get_bev(anchor)
            for k in args.k_list:
                bev_tk, top_tk = get_bev(anchor + k)
                stats = bev_pair_stats(bev_t, bev_tk)
                eg = ego_delta(
                    frame_assets[anchor]["ego2global"],
                    frame_assets[anchor]["lidar2ego"],
                    frame_assets[anchor + k]["ego2global"],
                    frame_assets[anchor + k]["lidar2ego"],
                )
                row = dict(
                    scene=args.scene, anchor_idx=anchor, k=k,
                    sample_t  = frame_assets[anchor]["sample_token"],
                    sample_tk = frame_assets[anchor + k]["sample_token"],
                    top1_score_t  = top_t["top1_score"],
                    top1_label_t  = top_t["top1_label"],
                    top1_score_tk = top_tk["top1_score"],
                    top1_label_tk = top_tk["top1_label"],
                    **{f"ego_{k_}": v for k_, v in eg.items()},
                    **stats,
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_written += 1
            print(f"[scene={args.scene}] anchor {ai+1}/{len(valid_anchors)} "
                  f"(idx={anchor}) took {time.time() - t_anchor:.2f}s; "
                  f"rows={n_written}", flush=True)

            # Drop bev_cache entries that no future anchor needs (keep memory low)
            if ai + 1 < len(valid_anchors):
                next_anchor = valid_anchors[ai + 1]
                keep = set(range(next_anchor - 0, next_anchor + max_k + 1))
                for kk in list(bev_cache.keys()):
                    if kk not in keep:
                        bev_cache.pop(kk, None)
                        top_cache.pop(kk, None)

    elapsed = time.time() - t_start
    print(f"[scene={args.scene}] done. wrote {n_written} rows in "
          f"{elapsed:.1f}s ({elapsed/max(1,n_written):.2f}s/row)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
