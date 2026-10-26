"""Fine-tune the age-conditioning port under delay augmentation.

Trains ONLY the ~350K-parameter port from [age_port.py](age_port.py); the
~60M-parameter detector is frozen. See that module for why the experiment
exists and what each outcome means.

WHY TRAIN THROUGH THE TEST-TIME PATH
------------------------------------
StreamPETR's own `forward_train` batches and shuffles sequences in a way that
would break the per-scene ring buffers our staleness machinery depends on
(`FeatureSubstitutor` counts frames within a scene; `HonestStaleness` keys its
pose buffer off the same counter). Instead this driver walks scenes frame by
frame exactly as inference does — `seq_mode=True` already forces
`queue_length=1`, so StreamPETR's sequential training is single-frame anyway —
and mirrors `simple_test_pts` step for step, swapping `get_bboxes` for
`pts_bbox_head.loss`. The head forward, the memory queue, and the scene resets
are therefore identical to deployment. Training conditions match the conditions
we measure in, which is the whole point of the experiment.

DELAY AUGMENTATION
------------------
Each scene is walked with a delay `k` sampled per scene from `--delay-choices`
(default 0..3, so the port sees fresh frames too and cannot simply learn to
ignore its input). Within a scene `k` is held fixed because the memory queue
carries state across frames: resampling mid-scene would train the port on
memory whose history was built under a different delay — the very
trajectory-mixing this project calls a chimera.

The honest key-side lift is always applied when k>0, so the port learns to
correct what is left after geometry is already right, not to compensate for a
coordinate error we introduced ourselves.

GRADIENT FLOW
-------------
The memory queue is detached by construction: `post_update_memory` calls
`.detach()` on everything it stores, so gradients never cross frame boundaries.
Each frame is an independent optimisation step over the current forward only.

Usage (see submit_train_age_port.sh for the SLURM wrapper):

    cd /ibex/user/khangea/multisensor/StreamPETR
    PYTHONPATH=$PWD:../stages/04_streampetr_latent:../stages/07_delay_conditioning \
    python ../stages/07_delay_conditioning/train_age_port.py \
        --config projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
        --ckpt ../checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth \
        --train-pkl ../data_withtrain/nuscenes2d_temporal_infos_train.pkl \
        --epochs 2 --out ../stages/07_delay_conditioning/dumps/age_port.pt
"""

import argparse
import os
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401 — register plugin classes
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from age_port import NUSC_DT, AgeConditioningPort, freeze_all_but_port, install_age_port

# ── Config ────────────────────────────────────────────────────────────────

N_CAMS = 6
DEFAULT_DELAYS = (0, 1, 2, 3)
# Kept small so even a 40-step smoke run prints several loss lines. A smoke test
# that verifies the plumbing but never shows the loss has not tested the thing
# that matters — whether the training signal is real.
LOG_EVERY = 10


# ── Data ──────────────────────────────────────────────────────────────────

def build_train_pipeline(cfg):
    """Deterministic image processing + GT loading.

    Takes the train pipeline's annotation steps but the test pipeline's image
    handling: `ResizeCropFlipRotImage(training=False)` and no
    `GlobalRotScaleTransImage`. Random flips/rotations would fight the
    experiment on two fronts — they perturb the ego-pose bookkeeping the honest
    protocol depends on, and they add variance we cannot afford with 62 scenes
    while measuring a small effect.
    """
    collect_keys = cfg.data.train.collect_keys
    ida = cfg.get("ida_aug_conf")
    class_names = cfg.class_names
    pc_range = cfg.point_cloud_range
    return [
        dict(type="LoadMultiViewImageFromFiles", to_float32=True),
        dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,
             with_bbox=True, with_label=True, with_bbox_depth=True),
        dict(type="ObjectRangeFilter", point_cloud_range=pc_range),
        dict(type="ObjectNameFilter", classes=class_names),
        dict(type="ResizeCropFlipRotImage", data_aug_conf=ida, training=False),
        dict(type="NormalizeMultiviewImage", **cfg.img_norm_cfg),
        dict(type="PadMultiViewImage", size_divisor=32),
        dict(type="PETRFormatBundle3D", class_names=class_names,
             collect_keys=[k for k in collect_keys if k not in ("img", "img_metas")]),
        # `union2one` unconditionally reads all six GT keys in non-test mode
        # ([nuscenes_dataset.py:138-143](../../StreamPETR/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py#L138-L143)),
        # so the 2D ones must be collected even though only the 3D loss is used
        # here — the frozen ROI head's 2D loss is never added to our objective.
        dict(type="Collect3D",
             keys=["gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels",
                   "centers2d", "depths", "img", "prev_exists"]
                  + [k for k in collect_keys if k not in ("img", "img_metas", "prev_exists")],
             meta_keys=("filename", "ori_shape", "img_shape", "pad_shape",
                        "scale_factor", "flip", "box_mode_3d", "box_type_3d",
                        "img_norm_cfg", "scene_token")),
    ]


def build_train_dataset(cfg, train_pkl):
    """Sequential, single-frame, GT-carrying dataset over the train scenes."""
    dcfg = cfg.data.train.copy()
    dcfg["ann_file"] = train_pkl
    dcfg["pipeline"] = build_train_pipeline(cfg)
    dcfg["seq_mode"] = True       # forces queue_length=1 -> one frame per item
    dcfg["seq_split_num"] = 1     # keep whole scenes intact; memory needs order
    dcfg["test_mode"] = False
    dcfg["filter_empty_gt"] = False
    return build_dataset(dcfg)


# ── Staleness driver ──────────────────────────────────────────────────────

class DelayDriver:
    """Per-scene feature ring buffer + the honest (T, dt) for the current frame.

    Deliberately standalone rather than reusing `FeatureSubstitutor` /
    `HonestStaleness`: those key their bookkeeping off a global sample counter
    matched to a precomputed scene-start list, which is a test-time construct.
    Here scene boundaries arrive directly in the data as `prev_exists == 0`,
    which is the same signal the head itself uses to reset memory, so the two
    can never disagree.
    """

    def __init__(self, model, delay_choices, rng):
        self.model = model
        self.delay_choices = list(delay_choices)
        self.rng = rng
        self.cache = [[] for _ in range(N_CAMS)]
        self.ego_pose, self.ego_pose_inv, self.lidar2img = [], [], []
        self.k = 0

    def new_scene(self):
        self.cache = [[] for _ in range(N_CAMS)]
        self.ego_pose, self.ego_pose_inv, self.lidar2img = [], [], []
        self.k = int(self.rng.choice(self.delay_choices))

    def step(self, data, img_feats):
        """Record this frame, substitute stale features, set the honest state.

        Args:
            data: the collated single-frame dict (ego_pose, lidar2img, ...).
            img_feats: fresh post-FPN features [1, N, C, H, W].

        Returns:
            (img_feats_used, k_applied) — features with per-camera staleness
            applied, and the delay actually in force (clamped at scene start).
        """
        for c in range(N_CAMS):
            self.cache[c].append(img_feats[0, c].detach())
        self.ego_pose.append(data["ego_pose"][0].detach().clone())
        self.ego_pose_inv.append(data["ego_pose_inv"][0].detach().clone())
        self.lidar2img.append(data["lidar2img"].detach().clone())

        f = len(self.ego_pose) - 1
        k = min(self.k, f)  # first k frames of a scene fall back to fresh

        if k > 0:
            out = img_feats.clone()
            for c in range(N_CAMS):
                out[0, c] = self.cache[c][f - k]
            # Honest key-side lift: un-project stale pixels at their true
            # capture pose, then express the result in the current lidar frame.
            data["lidar2img"] = self.lidar2img[f - k] @ (
                self.ego_pose_inv[f - k] @ self.ego_pose[f])
            self.model._age_state = {
                "T": self.ego_pose_inv[f] @ self.ego_pose[f - k],
                "dt": k * NUSC_DT,
            }
            return out, k

        self.model._age_state = {"T": None, "dt": 0.0}
        return img_feats, 0


# ── Train loop ────────────────────────────────────────────────────────────

def train_step(model, data, driver):
    """One frame: forward through the head, return the detection loss.

    Mirrors `Petr3D.simple_test_pts` up to the head output, then calls
    `pts_bbox_head.loss` instead of `get_bboxes`.
    """
    m = model.module
    # The loader hands us one flat dict, but StreamPETR's own forward keeps
    # img_metas as a POSITIONAL argument alongside `**data`
    # (`simple_test(img_metas, **data)`), so a dict that still holds an
    # "img_metas" key passes it twice — `prepare_location()` and the head then
    # raise "got multiple values for argument 'img_metas'". Same story for the
    # GT, which belongs to the loss rather than the forward. Split them out
    # here, on a shallow copy so the loader's dict is left alone.
    data = dict(data)
    img_metas = data.pop("img_metas")
    if isinstance(img_metas[0], list):
        img_metas = img_metas[0]
    gt_bboxes_3d = data.pop("gt_bboxes_3d")
    gt_labels_3d = data.pop("gt_labels_3d")

    is_new_scene = float(data["prev_exists"].sum()) == 0
    if is_new_scene:
        driver.new_scene()
        m.pts_bbox_head.reset_memory()

    with torch.no_grad():  # backbone/neck are frozen; no need for their graph
        img_feats = m.extract_img_feat(data["img"], 1)
    # Guard the eval-mode contract: 5-D [B, N, C, H, W]. If the model ever ends
    # up in train() this returns 6-D with a queue axis, and every downstream
    # index silently means something else.
    if img_feats.dim() != 5 or img_feats.size(1) != N_CAMS:
        raise RuntimeError(
            f"expected img_feats [B,{N_CAMS},C,H,W], got {tuple(img_feats.shape)}. "
            "Model must be in eval() mode (see the note in main()).")
    img_feats, k = driver.step(data, img_feats)
    data["img_feats"] = img_feats

    location = m.prepare_location(img_metas, **data)
    outs_roi = m.forward_roi_head(location, **data)
    topk_indexes = outs_roi["topk_indexes"]
    outs = m.pts_bbox_head(location, img_metas, topk_indexes, **data)

    losses = m.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
    # With the model in eval mode there are no denoising queries, so
    # `dn_mask_dict` is None and the head's `elif self.with_dn` branch fills the
    # `dn_*` entries with DETACHED duplicates of the main losses
    # ([streampetr_head.py:999-1010](../../StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L999-L1010)).
    # They carry no gradient; including them would only double-count the
    # reported loss and make the curve unreadable.
    total = sum(v for key, v in losses.items()
                if torch.is_tensor(v) and v.dim() == 0 and "dn_loss" not in key)
    return total, k


def unwrap(data, queue_keys):
    """Collapse the seq-mode collate into the shapes the head expects.

    `union2one` stacks every collect_key over the frame queue, so the loader
    yields `[B, T, ...]`. In seq_mode T is 1, and the head wants `[B, ...]` —
    the same shape `Petr3D.forward_test` produces for inference. Getting this
    wrong is silent: `[1, 1, 6, 4, 4]` would flow into `lidar2img.inverse()`
    and misinterpret the queue axis as cameras.

    GT keys arrive as nested lists rather than stacked tensors (they cannot be
    stacked — box counts differ per frame), so they are unwrapped to one list
    per batch element.
    """
    out = {}
    for key, val in data.items():
        if hasattr(val, "data"):
            val = val.data[0]
        if torch.is_tensor(val) and key in queue_keys and val.dim() >= 2:
            val = val[:, 0]  # drop the queue axis, keep the batch axis
        elif isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
            val = val[0]
        out[key] = val
    return out


def to_cuda(data):
    for key, val in data.items():
        if torch.is_tensor(val):
            data[key] = val.cuda()
        elif key in ("gt_bboxes_3d", "gt_labels_3d") and isinstance(val, list):
            data[key] = [v.to("cuda") if hasattr(v, "to") else v for v in val]
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-pkl", required=True)
    ap.add_argument("--out", required=True, help="where to save the trained port")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--delay-choices", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop early (smoke test); 0 = full epochs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    delays = [int(x) for x in args.delay_choices.split(",")]
    print(f"Delay choices (per scene): {delays}")

    cfg = Config.fromfile(args.config)
    # Do NOT null `cfg.model.train_cfg` here. The stage-04 eval scripts do, and
    # copying that was the bug: the head only builds `self.assigner` /
    # `self.sampler` when train_cfg is truthy
    # ([streampetr_head.py:149-158](../../StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L149-L158)),
    # and `loss()` needs both for Hungarian matching against GT.
    #
    # This config keeps train_cfg INSIDE the model dict (cfg.model.train_cfg),
    # not at top level — `cfg.get("train_cfg")` is None here, so passing it
    # would achieve nothing. Leaving cfg.model untouched is what does the work.
    #
    # Orthogonal to `model.eval()`: train_cfg decides which submodules exist,
    # eval() decides how the forward behaves. We want the assigner to exist AND
    # the forward to match deployment.
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])

    port = AgeConditioningPort(model.module.pts_bbox_head).cuda()
    n_frozen, n_train = freeze_all_but_port(model, port)
    install_age_port(model.module, port)
    print(f"Frozen params: {n_frozen:,} | trainable (port): {n_train:,} "
          f"({100 * n_train / (n_frozen + n_train):.3f}%)")

    dataset = build_train_dataset(cfg, args.train_pkl)
    scenes = {i["scene_token"] for i in dataset.data_infos}
    print(f"Train: {len(dataset)} samples across {len(scenes)} scenes")

    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    optimizer = torch.optim.AdamW(port.trainable_parameters(), lr=args.lr,
                                  weight_decay=0.01)
    driver = DelayDriver(model.module, delays, rng)
    queue_keys = {k for k in cfg.data.train.collect_keys if k != "img_metas"}

    # EVAL mode throughout, deliberately. `model.train()` would change the
    # forward in three ways that all break the premise of this experiment —
    # train under the conditions we measure in:
    #   * GridMask applies random image masking when `self.training`
    #     ([grid_mask.py:85](../../StreamPETR/projects/mmdet3d_plugin/models/utils/grid_mask.py#L85)),
    #     injecting noise into features from a backbone we are not training;
    #   * `prepare_for_dn` adds denoising queries, changing the query set the
    #     port conditions;
    #   * `extract_img_feat` returns a 6-D tensor with a queue axis instead of
    #     the 5-D inference shape.
    # Eval mode disables none of autograd, so the port still trains normally;
    # it only makes the forward identical to deployment.
    model.eval()

    step = 0
    k_seen = {}
    t0 = time.time()
    for epoch in range(args.epochs):
        run_loss, run_n = 0.0, 0
        for data in loader:
            data = to_cuda(unwrap(data, queue_keys))
            loss, k = train_step(model, data, driver)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(port.trainable_parameters(),
                                           args.grad_clip)
            optimizer.step()

            run_loss += float(loss)
            run_n += 1
            step += 1
            k_seen[k] = k_seen.get(k, 0) + 1
            if step % LOG_EVERY == 0:
                rate = step / (time.time() - t0)
                # Report the DISTRIBUTION of applied staleness, not just this
                # frame's k: if the port only ever saw k=0 the whole experiment
                # is vacuous, and a single trailing k cannot reveal that.
                hist = " ".join(f"k{kk}={v}" for kk, v in sorted(k_seen.items()))
                print(f"  epoch {epoch} step {step:>5}  loss {run_loss / run_n:.4f}  "
                      f"[{hist}]  {rate:.2f} it/s", flush=True)
                run_loss, run_n = 0.0, 0
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": port.state_dict(),
                "args": vars(args),
                "steps": step}, args.out)
    print(f"\nSaved port ({step} steps, {time.time() - t0:.0f}s) -> {args.out}")


if __name__ == "__main__":
    main()
