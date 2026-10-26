"""Add scene_token + frame_idx to the existing mmdet3d-v1 val pkl so that
MamBEV's NuScenesDatasetMultiFrame can use it.

The v1.x pkl at data_full/nuscenes_infos_val.pkl has per-sample fields
{sample_idx, token, timestamp, ego2global, images, lidar_points, instances,
cam_instances} but NOT scene_token. MamBEV's dataset hits
    cur_scene_token = example["data_samples"].scene_token
inside prepare_data() and crashes. The temporal multi-frame iterator at
nuscenes_dataset.py:247 (`for i, list_idx in zip(self.frames + index, self.frames)`)
also relies on scene_token to mask cross-scene neighbors.

Rather than re-running MamBEV's data converter (~40 min for partial blob), we
look each token up in nuScenes-devkit and add scene_token to the existing pkl.
We also add a within-scene frame_idx for completeness (MamBEV reads it as a
meta_key in mambev_small_t3.py:358).

Writes:  data_full/nuscenes_infos_temporal_val.pkl

Run once:

    conda activate /ibex/user/khangea/conda-environments/mambev_env
    python stages/06_gated_bev_psm/enrich_val_pkl.py

If MamBEV's pipeline crashes later on some other missing temporal field
(prev/next/lidar_sweeps/cam_sweeps) we fall back to running the full
converter — write a corresponding patched create_data.py at that point.
"""
import os
import pickle
import sys

from nuscenes import NuScenes


DATAROOT = "/ibex/user/khangea/multisensor/data_full"
SRC = os.path.join(DATAROOT, "nuscenes_infos_val.pkl")
DST = os.path.join(DATAROOT, "nuscenes_infos_temporal_val.pkl")


def main():
    # If a symlink was created earlier (Path 1 attempt), remove it first.
    if os.path.islink(DST):
        os.unlink(DST)
        print(f"Removed existing symlink at {DST}")

    print(f"Loading {SRC} ...")
    with open(SRC, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, dict) and "data_list" in data, \
        "Unexpected pkl schema — expected mmdet3d v1 with top-level data_list."

    print("Building token -> scene_token table from nuScenes-devkit ...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
    tok2scene = {s["token"]: s["scene_token"] for s in nusc.sample}

    print("Computing per-scene frame_idx (time-ordered within scene) ...")
    # Group by scene, sort by timestamp, assign frame_idx.
    by_scene = {}
    for info in data["data_list"]:
        st = tok2scene[info["token"]]
        by_scene.setdefault(st, []).append(info)
    for st, infos in by_scene.items():
        infos.sort(key=lambda i: i["timestamp"])
        for k, info in enumerate(infos):
            info["scene_token"] = st
            info["frame_idx"] = k

    n_scenes = len(by_scene)
    n_samples = len(data["data_list"])
    print(f"Enriched {n_samples} samples across {n_scenes} scenes.")

    print(f"Writing {DST} ...")
    with open(DST, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    main()
