"""
Build the 2D auxiliary labels StreamPETR's ROI head needs for SWEEP images (0 GPU).

StreamPETR trains a 2D head (boxes, centers2d, depths per camera) whose outputs
decide where 3D queries are spawned. The pkl only has these labels for keyframe
images. When a camera is swapped for an older 12 Hz sweep during delay
fine-tuning, the right 2D target is "where objects appear IN THAT IMAGE": the
devkit's annotations interpolated to the sweep's capture time, projected with
the sweep camera's own pose. (3D targets stay at keyframe time t — learning to
bring an old observation to the present is the point of the fine-tune.)

This mirrors the converter's `get_2d_boxes` + `_fill_trainval_infos` 2D block
exactly, swapping only the box source (`nusc.get_boxes(sd)`, which interpolates
for non-keyframes). Gate: on keyframes (j=0) it must reproduce the pkl's labels
bit-for-bit.

Usage (login node, ~10 min):
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/10_delay_finetune/build_sweep_labels2d.py
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

PROJECT_DIR = "/ibex/user/khangea/multisensor"
DATAROOT = os.path.join(PROJECT_DIR, "data_full")
TRAIN_PKL = os.path.join(PROJECT_DIR, "data_withtrain/nuscenes2d_temporal_infos_train.pkl")
SWEEP_INDEX = os.path.join(PROJECT_DIR, "stages/09_subframe_async/dumps/sweep_index_train.pkl")
OUT_PKL = os.path.join(PROJECT_DIR, "stages/10_delay_finetune/dumps/sweep_labels2d_train.pkl")
MAX_J = 3                 # fine-tune delays: 1..3 sweeps ≈ 80-250 ms
N_GATE_SAMPLES = 300      # keyframes checked against the pkl
IMG_W, IMG_H = 1600, 900  # nuScenes camera canvas (converter reads it per image; all equal)

sys.path.insert(0, os.path.join(PROJECT_DIR, "StreamPETR"))
from tools.data_converter.nuscenes_converter import (  # noqa: E402
    generate_record, nus_categories, post_process_coords)
from mmdet3d.core.bbox import points_cam2img  # noqa: E402
from nuscenes.utils.geometry_utils import view_points  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

LABEL_KEYS = ("bboxes2d", "labels2d", "centers2d", "depths", "bboxes_ignore")


def labels_2d(nusc, sd_token):
    """2D labels for one camera image, as the converter would build them.

    Args:
        nusc: NuScenes instance.
        sd_token: camera sample_data token (keyframe or sweep).

    Returns:
        dict with bboxes2d (N,4) f32 [x1,y1,x2,y2], labels2d (N,) int64,
        centers2d (N,2) f32, depths (N,) f32, bboxes_ignore (M,4) f32.
    """
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    K = np.array(cs["camera_intrinsic"])

    b2d, lab, c2d, dep, ign = [], [], [], [], []
    for box in nusc.get_boxes(sd_token):          # interpolated to sd time for sweeps
        ann_rec = dict(nusc.get("sample_annotation", box.token))
        ann_rec["sample_annotation_token"] = ann_rec["token"]
        ann_rec["sample_data_token"] = sd_token
        # global -> ego(sd time) -> camera
        box.translate(-np.array(pose["translation"]))
        box.rotate(Quaternion(pose["rotation"]).inverse)
        box.translate(-np.array(cs["translation"]))
        box.rotate(Quaternion(cs["rotation"]).inverse)

        corners = box.corners()
        corners = corners[:, np.argwhere(corners[2, :] > 0).flatten()]
        coords = post_process_coords(view_points(corners, K, True).T[:, :2].tolist())
        if coords is None:
            continue
        rec = generate_record(ann_rec, *coords, sd_token, sd["filename"])
        if rec is None:
            continue
        center2d = points_cam2img(np.array(box.center.tolist()).reshape(1, 3), K,
                                  with_depth=True).squeeze().tolist()
        if center2d[2] <= 0:
            continue

        # converter's per-camera filter (_fill_trainval_infos)
        x1, y1, w, h = rec["bbox"]
        inter_w = max(0, min(x1 + w, IMG_W) - max(x1, 0))
        inter_h = max(0, min(y1 + h, IMG_H) - max(y1, 0))
        if inter_w * inter_h == 0 or rec["area"] <= 0 or w < 1 or h < 1:
            continue
        if rec["category_name"] not in nus_categories:
            continue
        if rec.get("iscrowd", False):
            ign.append([x1, y1, x1 + w, y1 + h])
            continue
        b2d.append([x1, y1, x1 + w, y1 + h])
        lab.append(rec["category_id"])
        c2d.append(center2d[:2])
        dep.append(center2d[2])

    return dict(bboxes2d=np.array(b2d, dtype=np.float32),
                labels2d=np.array(lab, dtype=np.int64),
                centers2d=np.array(c2d, dtype=np.float32),
                depths=np.array(dep, dtype=np.float32),
                bboxes_ignore=np.array(ign, dtype=np.float32))


def gate(nusc, infos):
    """Keyframe labels from `labels_2d` must equal the pkl's, array for array."""
    worst = 0.0
    for info in infos[:: max(1, len(infos) // N_GATE_SAMPLES)]:
        for c, cam_info in enumerate(info["cams"].values()):
            got = labels_2d(nusc, cam_info["sample_data_token"])
            for k in LABEL_KEYS:
                ref = np.asarray(info[k][c])
                if ref.shape != got[k].shape:
                    print(f"[gate] shape mismatch {k} {ref.shape} vs {got[k].shape} "
                          f"at {info['token']} cam {c}")
                    return False
                if ref.size:
                    worst = max(worst, float(np.abs(ref.astype(np.float64) - got[k]).max()))
    print(f"[gate] keyframe labels vs pkl: max abs diff {worst:.2e}")
    return worst < 1e-3


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pkl", default=TRAIN_PKL)
    ap.add_argument("--sweep-index", default=SWEEP_INDEX)
    ap.add_argument("--out", default=OUT_PKL)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    from nuscenes.nuscenes import NuScenes
    t0 = time.time()
    nusc = NuScenes("v1.0-trainval", DATAROOT, verbose=False)
    print(f"NuScenes loaded in {time.time() - t0:.0f}s")

    infos = pickle.load(open(args.pkl, "rb"))["infos"]
    if not gate(nusc, infos):
        raise SystemExit("gate failed; not building")

    index = pickle.load(open(args.sweep_index, "rb"))["index"]
    out, t0, n_obj = {}, time.time(), np.zeros(MAX_J + 1)
    for n, info in enumerate(infos):
        per_cam = {}
        for cam in info["cams"]:
            chain = index[info["token"]][cam]
            per_cam[cam] = [None] + [
                labels_2d(nusc, chain[j]["sample_data_token"]) if chain[j] is not None else None
                for j in range(1, MAX_J + 1)]
            for j in range(1, MAX_J + 1):
                if per_cam[cam][j] is not None:
                    n_obj[j] += len(per_cam[cam][j]["labels2d"])
        out[info["token"]] = per_cam
        if (n + 1) % 200 == 0:
            rate = (n + 1) / (time.time() - t0)
            print(f"  {n + 1}/{len(infos)} samples  {rate:.1f}/s  "
                  f"ETA {(len(infos) - n - 1) / rate / 60:.1f} min")

    n_key = sum(len(l) for info in infos for l in info["labels2d"])
    print(f"2D objects: keyframes {n_key}, sweeps j=1..{MAX_J}: {n_obj[1:].astype(int).tolist()}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(max_j=MAX_J, labels=out), f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
