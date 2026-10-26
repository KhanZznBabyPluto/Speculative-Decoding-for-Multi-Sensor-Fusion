"""
StreamPETR training dataset with random per-camera sub-keyframe delay.

`StaleSweepNuScenesDataset` wraps `CustomNuScenesDataset`: after the parent builds
a training sample, each camera independently (prob `p_stale`) is swapped for its
j-th previous 12 Hz sweep, j ~ U{1..max_j} (≈ 80-250 ms for max_j=3), with
  * the sweep's honest extrinsics (camera pose at capture time in lidar(t)),
  * the sweep's own 2D ROI labels (objects as they appear in that image),
while the 3D targets stay at keyframe time t. With p_stale=0 it is exactly the
parent dataset — that is the control arm of the fine-tune.

Registered in mmdet's DATASETS; configs pull it in via
`custom_imports=dict(imports=['stale_dataset'])` with this directory on PYTHONPATH.
"""

import os
import pickle

import numpy as np
from mmdet.datasets import DATASETS

import projects.mmdet3d_plugin  # noqa: F401  (registers CustomNuScenesDataset)
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import (
    CustomNuScenesDataset, convert_egopose_to_matrix_numpy, invert_matrix_egopose_numpy)

DATAROOT = "/ibex/user/khangea/multisensor/data_full"
PKL_PATH_PREFIX = "./data/nuscenes/"


def sweep_lidar2img(entry, intrinsic_4x4):
    """(lidar2img (4,4), lidar2cam (4,4)) for a sweep entry, as get_data_info builds them."""
    cam2lidar = convert_egopose_to_matrix_numpy(entry["sensor2lidar_rotation"],
                                                entry["sensor2lidar_translation"])
    lidar2cam = invert_matrix_egopose_numpy(cam2lidar)
    return intrinsic_4x4 @ lidar2cam, lidar2cam


@DATASETS.register_module()
class StaleSweepNuScenesDataset(CustomNuScenesDataset):
    """CustomNuScenesDataset + random per-camera sweep substitution (train only).

    Args:
        sweep_index: pkl from stages/09 build_sweep_index.py (train split).
        labels2d: pkl from build_sweep_labels2d.py (2D labels per sweep).
        p_stale: per-camera probability of substituting a sweep.
        max_j: largest delay in sweep steps (must be <= both pkls' max_j).
    """

    def __init__(self, *args, sweep_index=None, labels2d=None, p_stale=0.5, max_j=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_stale = float(p_stale)
        self.max_j = int(max_j)
        if self.p_stale > 0:
            idx = pickle.load(open(sweep_index, "rb"))
            lab = pickle.load(open(labels2d, "rb"))
            assert self.max_j <= min(idx["max_j"], lab["max_j"]), (self.max_j, idx["max_j"], lab["max_j"])
            self.sweep_index, self.sweep_labels = idx["index"], lab["labels"]

    def _draw_delays(self, token, cams):
        """Per-camera delay in sweep steps (0 = keep keyframe), clamped at scene start."""
        delays = []
        for cam in cams:
            j = 0
            if np.random.rand() < self.p_stale:
                j = np.random.randint(1, self.max_j + 1)
                chain = self.sweep_index[token][cam]
                while j > 0 and (chain[j] is None or self.sweep_labels[token][cam][j] is None):
                    j -= 1
            delays.append(j)
        return delays

    def get_data_info(self, index):
        """Parent sample, then swap delayed cameras' image, geometry and 2D labels."""
        out = super().get_data_info(index)
        if self.test_mode or self.p_stale <= 0:
            return out
        info = self.data_infos[index]
        cams = list(info["cams"].keys())
        delays = self._draw_delays(info["token"], cams)
        if not any(delays):
            return out

        # copy every per-camera list we touch: the parent's lists alias data_infos
        for key in ("img_filename", "lidar2img", "extrinsics", "img_timestamp"):
            out[key] = list(out[key])
        ann = out["ann_info"]
        for key in ("bboxes", "labels", "centers2d", "depths", "bboxes_ignore"):
            ann[key] = list(ann[key])

        for c, (cam, j) in enumerate(zip(cams, delays)):
            if j == 0:
                continue
            e = self.sweep_index[info["token"]][cam][j]
            lab = self.sweep_labels[info["token"]][cam][j]
            out["img_filename"][c] = os.path.join(DATAROOT, os.path.relpath(e["data_path"], PKL_PATH_PREFIX))
            out["lidar2img"][c], out["extrinsics"][c] = sweep_lidar2img(e, out["intrinsics"][c])
            out["img_timestamp"][c] = e["timestamp"] / 1e6
            ann["bboxes"][c] = lab["bboxes2d"]
            ann["labels"][c] = lab["labels2d"]
            ann["centers2d"][c] = lab["centers2d"]
            ann["depths"][c] = lab["depths"]
            ann["bboxes_ignore"][c] = lab["bboxes_ignore"]
        out["stale_delays"] = delays
        return out
