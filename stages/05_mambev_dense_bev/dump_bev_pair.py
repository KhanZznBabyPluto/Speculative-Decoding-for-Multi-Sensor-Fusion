"""Stage 05 — Phase 2.1 (part 1 of 2): dump a BEV pair from real nuScenes data.

For the warp + retention test we need MamBEV's dense BEV grid (100×100×256)
captured at two consecutive frames `t` and `t+1` of the same scene. This
script:

  1. Picks one of the 23 partial-blob val scenes (default = first one).
  2. Finds 4 consecutive samples within that scene whose 6-cam files are
     all on disk (needed because `frames=[-2, -1, 0]` requires 3 prior
     frames before each anchor; we want anchors at t and t+1, so 4 frames
     total).
  3. For each anchor, loads 6 cam images via nuscenes-devkit, builds the
     `lidar2img` projections, runs `model.predict`, captures the BEV
     encoder's output via forward hook.
  4. Saves a single .pt with:
       - BEV_t, BEV_t1  (each shape (groups, 100*100, 256) — fp16)
       - ego2global_t, ego2global_t1   (4×4)
       - lidar2ego_t,  lidar2ego_t1    (4×4)
       - sample_tokens, scene_token
       - top-K detection outputs at each anchor (scores, labels, bboxes)

The analysis script (next file) loads this and computes the ego-motion
warp from BEV_t coords into BEV_t1 coords, then measures per-cell L2 and
top-K detection retention.

Defers detailed warp math and head re-invocation to the analysis script;
this file is just data acquisition.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT   = Path("/ibex/user/khangea/multisensor")
MAMBEV_ROOT = REPO_ROOT / "MamBEV"
CKPT_DIR    = REPO_ROOT / "checkpoints" / "mambev"
NUSC_ROOT   = REPO_ROOT / "data_full"

CONFIG = MAMBEV_ROOT / "projects/MAMBEV/configs/MamBEV/mambev_small_t3.py"
CKPT   = CKPT_DIR / "mambev_small_detector.pth"

# Post test-pipeline image dimensions:
#   raw 1600x900 → scale 0.8 → 1280x720 → pad to size_divisor=32 → 1280x736.
SRC_H, SRC_W = 900, 1600
SCALE        = 0.8
TGT_H, TGT_W = 736, 1280
N_CAMS       = 6
FRAMES       = (-2, -1, 0)
CAM_NAMES    = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK",  "CAM_BACK_LEFT",   "CAM_BACK_RIGHT",
)

IMG_MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)
IMG_STD  = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def setup_path() -> None:
    sys.path.insert(0, str(MAMBEV_ROOT))


# --------------------------------------------------------------------------- #
#  nuScenes helpers                                                           #
# --------------------------------------------------------------------------- #

def load_nuscenes():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)


def find_consecutive_samples(nusc, scene_token: str, n_needed: int) -> List[dict]:
    """Walk scene's sample chain, return first run of n_needed consecutive
    samples whose 6 cam files all exist on disk."""
    scene = nusc.get("scene", scene_token)
    cur = nusc.get("sample", scene["first_sample_token"])
    chain = [cur]
    while cur["next"]:
        cur = nusc.get("sample", cur["next"])
        chain.append(cur)

    def all_cams_on_disk(sample: dict) -> bool:
        for cam in CAM_NAMES:
            sd = nusc.get("sample_data", sample["data"][cam])
            if not (NUSC_ROOT / sd["filename"]).exists():
                return False
        return True

    for i in range(len(chain) - n_needed + 1):
        window = chain[i:i + n_needed]
        if all(all_cams_on_disk(s) for s in window):
            return window
    raise RuntimeError(f"No window of {n_needed} consecutive samples in scene "
                       f"{scene_token} has all 6 cams on disk.")


def quat_T_to_matrix(translation, rotation_wxyz) -> np.ndarray:
    from pyquaternion import Quaternion
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    M[:3, 3]  = np.asarray(translation, dtype=np.float64)
    return M


def compute_lidar2img(nusc, sample: dict) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Return (lidar2img per cam [4×4 @ resized intrinsics], ego2global_lidar,
    lidar2ego). Follows BEVFormer/MamBEV's convention."""
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs  = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_ego = nusc.get("ego_pose",          lidar_sd["ego_pose_token"])

    lidar2ego        = quat_T_to_matrix(lidar_cs["translation"],  lidar_cs["rotation"])
    ego2global_lidar = quat_T_to_matrix(lidar_ego["translation"], lidar_ego["rotation"])

    lidar2img_per_cam: List[np.ndarray] = []
    for cam in CAM_NAMES:
        cam_sd = nusc.get("sample_data", sample["data"][cam])
        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_eg = nusc.get("ego_pose",          cam_sd["ego_pose_token"])

        cam2ego_cam    = quat_T_to_matrix(cam_cs["translation"], cam_cs["rotation"])
        ego2global_cam = quat_T_to_matrix(cam_eg["translation"], cam_eg["rotation"])
        ego2cam        = np.linalg.inv(cam2ego_cam)
        global2ego_cam = np.linalg.inv(ego2global_cam)

        # lidar(t_lidar) → ego(t_lidar) → global → ego(t_cam) → cam(t_cam)
        lidar2cam = ego2cam @ global2ego_cam @ ego2global_lidar @ lidar2ego

        # K is 3×3 intrinsic; scale because image will be resized SCALE×
        K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
        K_resized = K.copy()
        K_resized[0, :] *= SCALE
        K_resized[1, :] *= SCALE
        K_pad = np.eye(4, dtype=np.float64)
        K_pad[:3, :3] = K_resized

        lidar2img_per_cam.append((K_pad @ lidar2cam).astype(np.float32))

    return lidar2img_per_cam, ego2global_lidar.astype(np.float32), lidar2ego.astype(np.float32)


def load_and_preprocess_imgs(nusc, sample: dict) -> np.ndarray:
    """Return (N_CAMS, 3, TGT_H, TGT_W) float32, normalized, BGR channel order,
    matching the test pipeline (scale 0.8, pad to /32, mean-subtract)."""
    import cv2
    out = np.zeros((N_CAMS, 3, TGT_H, TGT_W), dtype=np.float32)
    for i, cam in enumerate(CAM_NAMES):
        sd = nusc.get("sample_data", sample["data"][cam])
        path = str(NUSC_ROOT / sd["filename"])
        img = cv2.imread(path)  # BGR uint8, (900, 1600, 3)
        if img is None:
            raise RuntimeError(f"cv2.imread returned None for {path}")
        # scale 0.8 → (720, 1280)
        new_w, new_h = int(SRC_W * SCALE), int(SRC_H * SCALE)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # NormalizeMultiviewImage: img.astype(float32); subtract mean; divide std
        img = img.astype(np.float32)
        img = (img - IMG_MEAN[None, None, :]) / IMG_STD[None, None, :]
        # PadMultiViewImage size_divisor=32 (1280 already /32; 720 → 736 pad bottom)
        pad_h = TGT_H - new_h
        pad_w = TGT_W - new_w
        assert pad_w == 0 and pad_h >= 0, f"unexpected pad: w={pad_w} h={pad_h}"
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        # HWC → CHW
        out[i] = img.transpose(2, 0, 1)
    return out


# --------------------------------------------------------------------------- #
#  Det3DDataSample construction                                               #
# --------------------------------------------------------------------------- #

def build_data_sample(sample_token: str, frame_idx: int, scene_token: str,
                      lidar2img_list: List[np.ndarray],
                      ego2global_lidar: np.ndarray, lidar2ego: np.ndarray):
    from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes, Box3DMode
    eye4 = np.eye(4, dtype=np.float32)
    meta = dict(
        sample_idx       = sample_token,
        frame_idx        = frame_idx,
        scene_token      = scene_token,
        ego2global       = ego2global_lidar,
        lidar2ego        = lidar2ego,
        lidar2img        = lidar2img_list,
        lidar2cam        = [eye4.copy() for _ in range(N_CAMS)],  # head doesn't read this
        cam2img          = [eye4.copy() for _ in range(N_CAMS)],
        img_shape        = [(TGT_H, TGT_W, 3)] * N_CAMS,
        pad_shape        = [(TGT_H, TGT_W, 3)] * N_CAMS,
        ori_shape        = [(SRC_H, SRC_W, 3)] * N_CAMS,
        scale_factor     = SCALE,
        img_norm_cfg     = dict(mean=IMG_MEAN.tolist(), std=IMG_STD.tolist(), to_rgb=False),
        box_mode_3d      = Box3DMode.LIDAR,
        box_type_3d      = LiDARInstance3DBoxes,
        can_bus          = np.zeros(18, dtype=np.float32),
        prev_bev_exists  = False,
        timestamp        = float(frame_idx),
        mono_input_dict  = None,
        mono_ann_idx     = [],
        aug_param        = dict(CropResizeFlipImage_param=(0, 0, TGT_W, TGT_H, False)),
        crop_offset      = (0, 0),
        img_crop_offset  = (0, 0),
        filename         = [f"{cam}.jpg" for cam in CAM_NAMES],
        img_path         = [f"{cam}.jpg" for cam in CAM_NAMES],
        resize_img_shape = [(TGT_H, TGT_W, 3)] * N_CAMS,
        num_pts_feats    = 4,
        pts_filename     = sample_token + ".bin",
        axis_align_matrix = eye4.copy(),
        mask             = False,
    )
    ds = Det3DDataSample()
    ds.set_metainfo(meta)
    return ds


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def run_predict(model, hook_target, batch_inputs_dict, batch_data_samples, device):
    import torch
    captured = {}

    def hook(module, inputs, output):
        t = output if hasattr(output, "shape") else output[0]
        captured["bev"] = t.detach().to(torch.float16).cpu().clone()

    h = hook_target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.predict(batch_inputs_dict, batch_data_samples)
            torch.cuda.synchronize()
    finally:
        h.remove()
    return captured["bev"], out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=str,
                        default="265f002f02d447ad9074813292eef75e",
                        help="nuScenes scene token (default = first stage-04 scene)")
    parser.add_argument("--out",   type=Path,
                        default=Path("/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/bev_pair.pt"))
    args = parser.parse_args()

    setup_path()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scene token : {args.scene}")
    print(f"nuScenes    : {NUSC_ROOT}")
    print(f"Output      : {args.out}")

    # ---- nuScenes ----
    print("\n[1/4] Loading nuScenes metadata...")
    nusc = load_nuscenes()
    samples = find_consecutive_samples(nusc, args.scene, n_needed=4)
    print(f"  4 consecutive samples on disk: "
          f"{[s['token'][:8] for s in samples]}")

    # ---- Model ----
    print("\n[2/4] Building MamBEV model + loading checkpoint...")
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
    print(f"  model: {type(model).__name__}, BEV encoder at "
          f"pts_bbox_head.transformer.encoder ({type(bev_encoder).__name__})")

    # ---- Per-frame asset construction ----
    print("\n[3/4] Building per-frame inputs (images, lidar2img, ego pose)...")
    # We need 2 anchors (t and t+1). Each anchor needs frames -2, -1, 0
    # so we need a window of 4 consecutive samples [s0, s1, s2, s3] where:
    #   anchor t   = s2 with prev = [s0, s1]
    #   anchor t+1 = s3 with prev = [s1, s2]
    per_frame = {}
    for i, samp in enumerate(samples):
        t0 = time.time()
        imgs = load_and_preprocess_imgs(nusc, samp)
        lidar2img_list, ego2global, lidar2ego = compute_lidar2img(nusc, samp)
        per_frame[i] = dict(
            sample_token = samp["token"],
            imgs         = imgs,
            lidar2img    = lidar2img_list,
            ego2global   = ego2global,
            lidar2ego    = lidar2ego,
        )
        print(f"  sample {i} ({samp['token'][:8]}): imgs {imgs.shape}, "
              f"ego2global det {np.linalg.det(ego2global[:3,:3]):.4f}, "
              f"took {time.time()-t0:.2f}s")

    # ---- Two forwards: anchored at sample idx 2 (t) and 3 (t+1) ----
    print("\n[4/4] Running MamBEV.predict() twice (anchored at t and t+1)...")
    dump = dict(
        scene_token = args.scene,
        sample_tokens = [s["token"] for s in samples],
    )
    for anchor, key in [(2, "t"), (3, "t+1")]:
        prev_offsets = [(anchor - 2), (anchor - 1)]
        print(f"  anchor={anchor} ({samples[anchor]['token'][:8]}), "
              f"prev=[{samples[prev_offsets[0]]['token'][:8]}, "
              f"{samples[prev_offsets[1]]['token'][:8]}]")

        batch_inputs_dict  = {}
        batch_data_samples = {}
        for f, src_idx in zip(FRAMES, [prev_offsets[0], prev_offsets[1], anchor]):
            pf = per_frame[src_idx]
            t_im = torch.from_numpy(pf["imgs"]).unsqueeze(0).to(device)  # (1, N, 3, H, W)
            batch_inputs_dict[f]  = dict(img=t_im)
            batch_data_samples[f] = [build_data_sample(
                sample_token = pf["sample_token"],
                frame_idx    = f,
                scene_token  = args.scene,
                lidar2img_list = pf["lidar2img"],
                ego2global_lidar = pf["ego2global"],
                lidar2ego = pf["lidar2ego"],
            )]

        t0 = time.time()
        bev, results = run_predict(model, bev_encoder,
                                   batch_inputs_dict, batch_data_samples, device)
        dt = time.time() - t0
        sample = results[0]
        pi = sample.pred_instances_3d
        scores = pi.scores_3d.detach().cpu()
        labels = pi.labels_3d.detach().cpu()
        bboxes_tensor = pi.bboxes_3d.tensor.detach().cpu()
        topk = torch.topk(scores, k=min(20, scores.numel()))
        top_idx = topk.indices
        print(f"    bev shape={tuple(bev.shape)} dtype={bev.dtype}, "
              f"detections={scores.numel()}, "
              f"top-5 scores={[round(float(s), 3) for s in topk.values[:5]]}, "
              f"{dt:.2f}s")

        dump[f"bev_{key}"]         = bev
        dump[f"anchor_sample_{key}"] = samples[anchor]["token"]
        dump[f"ego2global_{key}"]  = per_frame[anchor]["ego2global"]
        dump[f"lidar2ego_{key}"]   = per_frame[anchor]["lidar2ego"]
        dump[f"top20_scores_{key}"] = topk.values
        dump[f"top20_labels_{key}"] = labels[top_idx]
        dump[f"top20_bboxes_{key}"] = bboxes_tensor[top_idx]

    torch.save(dump, args.out)
    print(f"\nSaved dump → {args.out}")
    print(f"  approx size: {os.path.getsize(args.out) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
