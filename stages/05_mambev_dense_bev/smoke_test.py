"""Stage 05 — Phase 0.2 (lite): synthetic full-forward smoke test.

Goal: confirm the model runs end-to-end on CUDA (backbone → MamBEV encoder
→ BEV grid → DETR-style decoder → detections) on a fully synthetic batch,
*without* needing nuScenes data prep. This catches:

  * patches accidentally breaking the model's forward graph,
  * mmcv-2.1 + torch 2.3 + mamba-ssm runtime kernel issues that don't
    surface during weight-loading,
  * shape contract violations between encoder and decoder.

What this does NOT prove:
  * mAP, NDS, calibration — those need real nuScenes data + GT (Phase 0.2
    full, deferred).
  * That the speculation hypothesis works — that's Phase 2.1.

Output:
  * timings per stage (backbone × N frames, full predict)
  * BEV grid shape captured via forward_hook on pts_bbox_head.transformer.encoder
  * detection count + top-5 scores per class on the 0-frame.

Run from the activated mambev_env on a GPU node:
    python stages/05_mambev_dense_bev/smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

REPO_ROOT   = Path("/ibex/user/khangea/multisensor")
MAMBEV_ROOT = REPO_ROOT / "MamBEV"
CKPT_DIR    = REPO_ROOT / "checkpoints" / "mambev"

CONFIG = MAMBEV_ROOT / "projects/MAMBEV/configs/MamBEV/mambev_small_t3.py"
CKPT   = CKPT_DIR / "mambev_small_detector.pth"

# After mambev_small_t3 test_pipeline: scale 0.8 of 1600x900 → 1280x720,
# pad to size_divisor=32 → 1280x736.
IMG_H, IMG_W = 736, 1280
N_CAMS       = 6
FRAMES       = (-2, -1, 0)


def setup_path() -> None:
    sys.path.insert(0, str(MAMBEV_ROOT))


def banner(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def section(s: str) -> None:
    print(f"\n--- {s} ---")


def synth_img_meta(scene_token: str, frame_idx: int, device) -> dict:
    """Minimal img_meta that satisfies what BEVFormerHead_GroupDETRV3 reads.

    Real values come from the test pipeline; we fabricate plausibly-shaped
    placeholders. Identity projections won't produce sensible detections but
    won't crash the math, which is what this smoke test cares about.
    """
    import numpy as np
    # mmdet3d 1.4: `box_type_3d` in img_metas must be the class itself,
    # not the string "LiDAR" — predict_by_feat calls it as a constructor.
    from mmdet3d.structures import LiDARInstance3DBoxes, Box3DMode
    eye4 = np.eye(4, dtype=np.float32)
    return dict(
        sample_idx     = f"{scene_token}_f{frame_idx}",
        frame_idx      = frame_idx,
        scene_token    = scene_token,
        ego2global     = eye4.copy(),
        lidar2ego      = eye4.copy(),
        lidar2img      = [eye4.copy() for _ in range(N_CAMS)],
        lidar2cam      = [eye4.copy() for _ in range(N_CAMS)],
        cam2img        = [eye4.copy() for _ in range(N_CAMS)],
        img_shape      = [(IMG_H, IMG_W, 3)] * N_CAMS,
        pad_shape      = [(IMG_H, IMG_W, 3)] * N_CAMS,
        ori_shape      = [(900, 1600, 3)] * N_CAMS,
        scale_factor   = 0.8,
        img_norm_cfg   = dict(mean=[103.530, 116.280, 123.675],
                              std=[1.0, 1.0, 1.0], to_rgb=False),
        box_mode_3d    = Box3DMode.LIDAR,
        box_type_3d    = LiDARInstance3DBoxes,
        can_bus        = np.zeros(18, dtype=np.float32),
        prev_bev_exists = False,
        timestamp      = float(frame_idx),
        mono_input_dict = None,
        mono_ann_idx    = [],
        # used by GridMask + augmentation paths
        aug_param      = dict(CropResizeFlipImage_param=(0, 0, IMG_W, IMG_H, False)),
        crop_offset    = (0, 0),
        img_crop_offset = (0, 0),
        filename       = [f"synthetic_cam{c}.jpg" for c in range(N_CAMS)],
        img_path       = [f"synthetic_cam{c}.jpg" for c in range(N_CAMS)],
        resize_img_shape = [(IMG_H, IMG_W, 3)] * N_CAMS,
        num_pts_feats  = 4,
        pts_filename   = "synthetic.bin",
        axis_align_matrix = eye4.copy(),
        # mask flag the model checks during pack
        mask           = False,
    )


def build_data_sample(meta: dict):
    """Wrap an img_meta dict into a Det3DDataSample via mmengine."""
    from mmdet3d.structures import Det3DDataSample
    s = Det3DDataSample()
    s.set_metainfo(meta)
    return s


def main() -> int:
    setup_path()

    banner("Stage 05 / Phase 0.2 lite — synthetic full-forward smoke test")
    print(f"  config : {CONFIG}")
    print(f"  ckpt   : {CKPT}")
    print(f"  device : cuda (3-frame batch, {N_CAMS} cams, {IMG_H}×{IMG_W})")

    # ---- Imports / registration ------------------------------------------
    section("(1) Register MamBEV modules + build model")
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import projects.MAMBEV.mambev  # noqa: F401  (triggers MODELS.register)

    from mmengine.config import Config
    from mmengine.registry import MODELS
    from mmengine.runner import load_checkpoint

    cfg   = Config.fromfile(str(CONFIG))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(CKPT), map_location="cpu", strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  built: {type(model).__name__}, {n_params/1e6:.2f}M params")

    import torch
    if not torch.cuda.is_available():
        print("  FAIL — no CUDA available; can't run mamba-ssm kernels on CPU.")
        return 1
    device = torch.device("cuda")
    model = model.to(device).eval()

    # ---- BEV-shape hook --------------------------------------------------
    section("(2) Hook BEV encoder to capture its output shape")
    captured = {}

    def bev_hook(module, inputs, output):
        # The perception-transformer encoder returns a tensor of shape
        # (bev_h * bev_w, B, D) or similar. Stash shape only — no copy.
        t = output if isinstance(output, torch.Tensor) else output[0]
        captured.setdefault("bev_encoder_out_shape", tuple(t.shape))
        captured.setdefault("bev_encoder_out_dtype", str(t.dtype))

    bev_encoder = model.pts_bbox_head.transformer.encoder
    h = bev_encoder.register_forward_hook(bev_hook)

    # ---- Build synthetic 3-frame batch -----------------------------------
    section("(3) Build synthetic 3-frame batch")
    scene_token = "smoke_test_scene_0"
    batch_inputs_dict = {}
    batch_data_samples = {}
    for f in FRAMES:
        imgs = torch.randn(1, N_CAMS, 3, IMG_H, IMG_W,
                           device=device, dtype=torch.float32)
        batch_inputs_dict[f]  = dict(img=imgs)
        batch_data_samples[f] = [build_data_sample(synth_img_meta(scene_token, f, device))]
    print(f"  frames keys     : {sorted(batch_inputs_dict.keys())}")
    print(f"  current-frame img: {tuple(batch_inputs_dict[0]['img'].shape)} "
          f"{batch_inputs_dict[0]['img'].dtype}")

    # ---- Forward ---------------------------------------------------------
    section("(4) model.predict() on the synthetic batch")
    t0 = time.time()
    try:
        with torch.no_grad():
            results = model.predict(batch_inputs_dict, batch_data_samples)
        torch.cuda.synchronize()
    except Exception as exc:
        h.remove()
        print(f"  FAIL — {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return 1
    h.remove()
    t1 = time.time()
    print(f"  wall time       : {(t1 - t0):.2f} s")

    # ---- Report ----------------------------------------------------------
    section("(5) Captured BEV shape + detection structure")
    for k, v in captured.items():
        print(f"  {k:<28} {v}")

    if not isinstance(results, list) or not results:
        print(f"  FAIL — predict returned {type(results).__name__}, expected non-empty list")
        return 1
    sample = results[0]
    print(f"  pred sample type            : {type(sample).__name__}")

    if hasattr(sample, "pred_instances_3d"):
        pi = sample.pred_instances_3d
        scores = getattr(pi, "scores_3d", None)
        labels = getattr(pi, "labels_3d", None)
        bboxes = getattr(pi, "bboxes_3d", None)
        n = int(scores.numel()) if scores is not None else 0
        print(f"  pred_instances_3d count     : {n}")
        if n > 0 and scores is not None:
            topk = min(5, n)
            top_scores, top_idx = torch.topk(scores, topk)
            top_labels = labels[top_idx] if labels is not None else None
            print(f"  top-{topk} scores             : "
                  f"{[round(float(s), 4) for s in top_scores]}")
            if top_labels is not None:
                print(f"  top-{topk} labels             : {top_labels.tolist()}")
            if bboxes is not None:
                print(f"  bboxes_3d type              : {type(bboxes).__name__}")
                if hasattr(bboxes, "tensor"):
                    print(f"  bboxes tensor shape         : {tuple(bboxes.tensor.shape)}")
    else:
        print("  WARN — sample has no pred_instances_3d attribute")

    banner("PASS — synthetic full-forward completed on CUDA")
    print("Next: Phase 2.1 — same forward, but on a real 2-frame nuScenes pair,")
    print("      with the second frame's BEV warped by ego motion. Measures whether")
    print("      the head retains heatmap mass under stale-warped BEV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
