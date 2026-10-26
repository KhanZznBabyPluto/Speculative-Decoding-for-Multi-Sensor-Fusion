"""
Index the 12 Hz camera sweeps behind every val keyframe (0 GPU).

For each sample in the StreamPETR val pkl, each of the 6 cameras, and each
j = 0..MAX_J, follow the camera's sample_data `prev` chain j steps back and
record the older image's path, capture timestamp, and its HONEST camera->lidar(t)
extrinsics. "Honest" = the converter's own `obtain_sensor2top` evaluated on the
sweep's sample_data: the camera pose at the sweep's capture time, expressed in
the keyframe's lidar frame. So feeding (sweep image, honest extrinsics) is a
stale photo with its true capture geometry; (sweep image, keyframe extrinsics)
is the naive condition.

Gate: j=0 must reproduce the pkl's own path and sensor2lidar exactly — it is the
same function on the same sample_data.

Usage (login node is fine, ~2-3 min, mostly NuScenes metadata load):
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/09_subframe_async/build_sweep_index.py
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_DIR = "/ibex/user/khangea/multisensor"
DATAROOT = os.path.join(PROJECT_DIR, "data_full")
VAL_PKL = os.path.join(DATAROOT, "nuscenes2d_temporal_infos_val.pkl")
OUT_PKL = os.path.join(PROJECT_DIR, "stages/09_subframe_async/dumps/sweep_index_val.pkl")
PKL_PATH_PREFIX = "./data/nuscenes/"   # how the pkl spells image paths (relative to StreamPETR/)
MAX_J = 6                              # 6 sweeps ≈ 500 ms ≈ one keyframe interval

sys.path.insert(0, os.path.join(PROJECT_DIR, "StreamPETR"))
from tools.data_converter.nuscenes_converter import obtain_sensor2top  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402


def keyframe_lidar_calib(info):
    """(l2e_t, l2e_r_mat, e2g_t, e2g_r_mat) of the keyframe lidar, as the converter builds them."""
    return (np.array(info["lidar2ego_translation"]),
            Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
            np.array(info["ego2global_translation"]),
            Quaternion(info["ego2global_rotation"]).rotation_matrix)


def to_pkl_path(abs_path):
    """Rewrite an absolute dataroot path into the pkl's './data/nuscenes/...' form."""
    return PKL_PATH_PREFIX + os.path.relpath(abs_path, DATAROOT)


def index_camera(nusc, sd_token, calib):
    """Walk one camera's sample_data chain back MAX_J steps.

    Args:
        nusc: NuScenes instance.
        sd_token: sample_data token of the keyframe image for this camera.
        calib: keyframe lidar calibration from `keyframe_lidar_calib`.

    Returns:
        list of length MAX_J+1; entry j is a dict (data_path, timestamp [us],
        sensor2lidar_rotation (3,3), sensor2lidar_translation (3,),
        sample_data_token, is_key_frame) or None once the chain runs out
        (scene start).
    """
    entries, tok = [], sd_token
    for _ in range(MAX_J + 1):
        if not tok:
            entries.append(None)
            continue
        sweep = obtain_sensor2top(nusc, tok, *calib, sensor_type="camera")
        sd = nusc.get("sample_data", tok)
        entries.append(dict(
            data_path=to_pkl_path(sweep["data_path"]),
            timestamp=sweep["timestamp"],
            sensor2lidar_rotation=sweep["sensor2lidar_rotation"],
            sensor2lidar_translation=sweep["sensor2lidar_translation"],
            sample_data_token=tok,
            is_key_frame=sd["is_key_frame"],
        ))
        tok = sd["prev"]
    return entries


def gate_and_report(infos, index):
    """Check j=0 against the pkl and print sweep-age / file-existence statistics."""
    max_dr = max_dt = 0.0
    path_mismatch = missing = 0
    ages = defaultdict(list)          # j -> list of (keyframe cam ts - sweep ts) in ms
    key_at = defaultdict(int)         # j -> how often entry j is itself a keyframe
    for info in infos:
        for cam, cam_info in info["cams"].items():
            chain = index[info["token"]][cam]
            e0 = chain[0]
            max_dr = max(max_dr, np.abs(e0["sensor2lidar_rotation"] - cam_info["sensor2lidar_rotation"]).max())
            max_dt = max(max_dt, np.abs(e0["sensor2lidar_translation"] - cam_info["sensor2lidar_translation"]).max())
            # basename only: the regenerated train pkl spells paths absolutely via data_withtrain/
            path_mismatch += os.path.basename(e0["data_path"]) != os.path.basename(cam_info["data_path"])
            for j, e in enumerate(chain):
                if e is None:
                    continue
                ages[j].append((e0["timestamp"] - e["timestamp"]) / 1e3)
                key_at[j] += bool(e["is_key_frame"])
                if not os.path.isfile(os.path.join(DATAROOT, os.path.relpath(e["data_path"], PKL_PATH_PREFIX))):
                    missing += 1

    print(f"\n[gate] j=0 vs pkl: max |dR|={max_dr:.2e}  max |dt|={max_dt:.2e} m  "
          f"path mismatches={path_mismatch}")
    print(f"[files] missing sweep images: {missing}")
    print("\n  j   n       age ms: mean   min    max    | keyframe share")
    for j in range(MAX_J + 1):
        a = np.array(ages[j])
        print(f"  {j}  {len(a):5d}  {a.mean():8.1f} {a.min():6.1f} {a.max():6.1f}   | "
              f"{key_at[j] / max(len(a), 1):.2f}")
    ok = max_dr < 1e-9 and max_dt < 1e-9 and path_mismatch == 0 and missing == 0
    print(f"\n[gate] {'PASS' if ok else 'FAIL'}")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val-pkl", default=VAL_PKL)
    ap.add_argument("--out", default=OUT_PKL)
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    from nuscenes.nuscenes import NuScenes
    t0 = time.time()
    nusc = NuScenes("v1.0-trainval", DATAROOT, verbose=False)
    print(f"NuScenes loaded in {time.time() - t0:.0f}s")

    infos = pickle.load(open(args.val_pkl, "rb"))["infos"]
    index = {}
    t0 = time.time()
    for n, info in enumerate(infos):
        calib = keyframe_lidar_calib(info)
        index[info["token"]] = {cam: index_camera(nusc, ci["sample_data_token"], calib)
                                for cam, ci in info["cams"].items()}
        if (n + 1) % 200 == 0:
            print(f"  indexed {n + 1}/{len(infos)} samples  ({(n + 1) / (time.time() - t0):.0f}/s)")

    if not gate_and_report(infos, index):
        raise SystemExit("gate failed; not writing index")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(max_j=MAX_J, index=index), f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
