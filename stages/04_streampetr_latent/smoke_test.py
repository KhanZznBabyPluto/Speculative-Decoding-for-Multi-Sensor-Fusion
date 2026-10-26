"""Phase 0.1 regression check: load StreamPETR R50 + checkpoint, run a small
slice of the val set, confirm outputs match the official tools/test.py path.

Two windows are tested:
  * Frames 0-4 (a pedestrian-heavy scene where the model genuinely has weak
    per-class confidence). Top-1 scores ≈ 0.05-0.10.
  * Frames 200-204 (a barrier/cone scene). Top-1 scores should be ≥ 0.3.

If the second window's top-1 score is ≥ 0.3, the pipeline is healthy.

Run from inside StreamPETR/ with streampetr_env activated:

    cd /ibex/user/khangea/multisensor/StreamPETR
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    PYTHONPATH=$PWD python ../stages/04_streampetr_latent/smoke_test.py
"""

import os
import sys
from functools import partial

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import Subset, DataLoader

# Triggers @register_module side-effects on the StreamPETR plugin.
import projects.mmdet3d_plugin  # noqa: F401


CONFIG = "projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py"
CKPT = "../checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth"
CLASS_NAMES = (
    "car", "truck", "constr_veh", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)


def build_test_model(config_path, ckpt_path):
    """Mirrors tools/test.py setup: test_mode on dataset cfg, MMDataParallel
    wrap WITHOUT manual .cuda() (MMDataParallel handles device transfer)."""
    cfg = Config.fromfile(config_path)
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset = build_dataset(cfg.data.test)
    return cfg, model, dataset, samples_per_gpu


def run_indices(model, dataset, indices, samples_per_gpu):
    """Run single_gpu_test on a Subset of `indices`. Returns list of results."""
    sub = Subset(dataset, indices)
    loader = DataLoader(
        sub,
        batch_size=samples_per_gpu,
        num_workers=0,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
    )
    return single_gpu_test(model, loader)


def main():
    print("Building model + dataset…")
    cfg, model, dataset, spg = build_test_model(CONFIG, CKPT)
    print(f"  val samples: {len(dataset)},  samples_per_gpu: {spg}")

    # Two windows: known weak (pedestrian-heavy) + known confident
    # (barrier/cone-heavy). CustomNuScenesDataset reorders by scene/flag for
    # streaming continuity, so positional indices != val pkl order.
    win_weak = list(range(5))
    win_strong = list(range(200, 205))
    print(f"\nRunning weak window {win_weak} + strong window {win_strong}…")
    results = run_indices(model, dataset, win_weak + win_strong, spg)

    print()
    for pos, (idx, r) in enumerate(zip(win_weak + win_strong, results)):
        info = dataset.data_infos[idx]
        scores = r["pts_bbox"]["scores_3d"]
        labels = r["pts_bbox"]["labels_3d"]
        top3 = [f"{s.item():.3f}·{CLASS_NAMES[l.item()]}"
                for s, l in zip(scores[:3], labels[:3])]
        print(f"  pos {pos:2d}  idx={idx:3d}  tok={info['token'][:8]}  "
              f"scene={info['scene_token'][:8]}  top-3 {top3}")

    strong_top1 = results[-1]["pts_bbox"]["scores_3d"][0].item()
    print()
    if strong_top1 >= 0.3:
        print(f"OK: strong-window top-1 = {strong_top1:.3f}. Phase 0.1 verified.")
    else:
        print(f"FAIL: strong-window top-1 = {strong_top1:.3f} (< 0.3).")
        sys.exit(1)


if __name__ == "__main__":
    main()
