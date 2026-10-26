"""Honest-protocol staleness: tell StreamPETR WHEN and WHERE its photons landed.

WHY THIS EXISTS
---------------
Our staleness harness (measure_stale_map.py) substitutes only the post-FPN image
features from frame t-k. Every piece of metadata stays at frame t. Verified in
source:

  - streampetr_head.py:402  `img2lidars = data['lidar2img'].inverse()`
        The stale pixels are lifted into the ego frame at t-k, but the head
        reads the resulting 3D points as ego-frame-at-t coordinates.
  - streampetr_head.py:426  `rec_ego_pose = torch.eye(4)`          (unconditional)
  - streampetr_head.py:438  `time_embedding(pos2posemb1d(zeros))`  (unconditional)
        A query born from stale photons is told "you are here, now".

This is NOT a bug in StreamPETR. In fresh operation `(identity, dt=0)` is
correct by definition. It is an unconditional SYNCHRONY ASSUMPTION: the
architecture has no port through which "this observation is old" can be
expressed for image-born queries. A naive deployment wrapper around a
synchrony-trained model therefore does exactly what our harness does.

stratify_by_egomotion.py showed the consequence is not academic. At delay=1,
mAP retention vs fresh is 85.8% on near-static frames but 11.5% when the ego
moved >4 m. Traffic cones -- which cannot move -- retain 97.5% when the ego is
static and 30.1% when it is not, so the damage is ego-induced, not object
motion.

WHAT THE HONEST PROTOCOL DOES
-----------------------------
A real asynchronous camera stack delivers a frame captured at t-k together with
its capture timestamp and (via 100-200 Hz odometry) its capture pose. So the
honest condition is not "hide the staleness" but "declare it". Two edits, both
zero-parameter, using only information a real system has:

  KEY SIDE (the 3D lift). The head computes img2lidar = inv(data['lidar2img']).
  We want stale pixels un-projected at the pose where they were captured, then
  expressed in the current lidar frame:

      img2lidar_honest = T_{lidar(t)<-lidar(t-k)} @ inv(lidar2img_{t-k})
  =>  lidar2img_used   = lidar2img_{t-k} @ T_{lidar(t-k)<-lidar(t)}

  where T_{lidar(t-k)<-lidar(t)} = ego_pose_inv(t-k) @ ego_pose(t).
  Note this is NOT "swap in the t-k calibration": lidar2img is a rigid
  calibration and is ~identical at t and t-k, which is exactly why the error
  goes uncorrected. The correction is the composition with ego motion.

  QUERY SIDE (the conditioning). Replace the hardcoded (identity, dt=0) with
  (T_{lidar(t)<-lidar(t-k)}, dt = k*0.5 s).

  This is elegantly IN-DISTRIBUTION for the pretrained weights. `rec_ego_motion`
  is cat([zeros(3), pose[:3,:].flatten()]) -> 15 dims -> the SAME `MLN` module
  the memory path feeds with cat([memory_velo(2), memory_timestamp(1),
  memory_egopose(12)]). Those three zeros sit exactly in the [velo, velo,
  timestamp] slot. Writing [0, 0, k*0.5] with a real relative pose is the very
  format the shared MLN was trained on via the carried-memory path -- we are
  handing already-trained machinery an input it already understands.

  Memory bookkeeping follows for free: post_update_memory stamps
  memory_egopose = data['ego_pose'] @ rec_ego_pose, and with
  rec_ego_pose = T_{lidar(t)<-lidar(t-k)} that yields ego_pose(t-k) -- literally
  "the pose where these photons were captured". Only rec_timestamp needs a
  nudge (it is hardcoded to 0), applied after the call.

WHAT IS AND IS NOT CORRECTED
----------------------------
Corrected: the ego-induced coordinate error (mean 3.09 m per 0.5 s frame here).
NOT corrected: object motion during the staleness gap. That residual is the
genuine, irreducible information loss of running late -- and it is what a
learned or analytical mitigation would have to address. Separating the two is
the point.

MODES (--honest-mode) let the two halves be attributed independently:
    none   -- reproduce the naive harness exactly (sanity check)
    key    -- lift correction only
    query  -- conditioning correction only
    both   -- the full honest protocol

Self-test: at --delay-all 0 every mode MUST be a bit-exact no-op (the applied
delay is 0, so T = I and dt = 0), reproducing the fresh baseline mAP 0.4448.

Usage:

    cd /ibex/user/khangea/multisensor/StreamPETR
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    PYTHONPATH=$PWD python ../stages/04_streampetr_latent/honest_staleness.py \
        --config projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
        --ckpt   ../checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth \
        --delay-all 1 --honest-mode both \
        --out    ../stages/04_streampetr_latent/dumps/honest_d1_both
"""

import argparse
import os
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401 — register plugin classes
# The plugin's build_dataloader is the one that accepts `nonshuffler_sampler`
# (the sequential sampler streaming inference needs); mmdet3d's does not.
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from measure_stale_map import (FeatureSubstitutor, _compute_scene_starts,
                               _install_partial_blob_eval_patch, _parse_delay)

# ── Config ────────────────────────────────────────────────────────────────

NUSC_DT = 0.5  # nuScenes keyframe spacing (2 Hz), seconds
HONEST_MODES = ("none", "key", "query", "both")


# ── Honest conditioning ───────────────────────────────────────────────────

class HonestStaleness:
    """Feeds the head the true capture pose and age of stale observations.

    Installs three patches on a loaded StreamPETR model:

      1. `StreamPETRHead.forward` wrapper -- maintains per-scene ring buffers of
         ego pose and lidar2img, computes this frame's honest (T, dt) from the
         delay the FeatureSubstitutor actually applied, and (in 'key'/'both'
         mode) rewrites data['lidar2img'] before position_embeding runs.
      2. `temporal_alignment` replacement -- (in 'query'/'both' mode) conditions
         the newly-born queries on (T, dt) instead of (identity, 0).
      3. `post_update_memory` wrapper -- ages the new block's memory_timestamp
         by dt, since rec_timestamp is hardcoded to zero upstream.

    Args:
        model: the MMDataParallel-wrapped detector.
        substitutor: the live FeatureSubstitutor (its `last_applied_delay` is
            the authority on how stale this frame actually is, already clamped
            to the cache length at scene starts).
        mode: one of HONEST_MODES.
    """

    def __init__(self, model, substitutor, mode):
        if mode not in HONEST_MODES:
            raise ValueError(f"mode must be one of {HONEST_MODES}")
        self.mode = mode
        self.substitutor = substitutor
        self.head = model.module.pts_bbox_head

        # Per-scene ring buffers, indexed by frame-in-scene.
        self.ego_pose = []
        self.ego_pose_inv = []
        self.lidar2img = []

        self.frame_in_scene = 0
        self._honest_T = None    # [4,4] lidar(t) <- lidar(t-k)
        self._honest_dt = 0.0

        self._orig_forward = self.head.forward
        self._orig_temporal_alignment = self.head.temporal_alignment
        self._orig_post_update = self.head.post_update_memory

        self.head.forward = self._forward_hook
        if mode in ("query", "both"):
            self.head.temporal_alignment = self._honest_temporal_alignment
            self.head.post_update_memory = self._post_update_hook

    # -- frame bookkeeping -------------------------------------------------

    def _reset(self):
        self.ego_pose, self.ego_pose_inv, self.lidar2img = [], [], []
        self.frame_in_scene = 0

    def _forward_hook(self, memory_center, img_metas, topk_indexes=None, **data):
        """Compute this frame's honest (T, dt); optionally fix the 3D lift."""
        # prev_exists == 0 marks a scene boundary (set in simple_test_pts).
        if float(data["prev_exists"].sum()) == 0:
            self._reset()

        self.ego_pose.append(data["ego_pose"][0].clone())
        self.ego_pose_inv.append(data["ego_pose_inv"][0].clone())
        self.lidar2img.append(data["lidar2img"].clone())
        f = len(self.ego_pose) - 1

        # The substitutor already applied (and clamped) the delay for this
        # frame's features; trust it rather than recomputing the clamp. Its
        # frame counter is incremented inside extract_img_feat, which runs
        # earlier in the same sample, so it should be exactly one ahead of ours.
        # A mismatch would mean our pose buffer indexes a different frame than
        # the features came from -- silently wrong, so fail loudly instead.
        expected = self.substitutor.frame_in_scene - 1
        if expected != f:
            raise RuntimeError(
                f"frame desync: honest buffer at {f}, substitutor at {expected}. "
                "Scene-boundary detection disagrees between the two hooks.")

        k = int(self.substitutor.last_applied_delay)
        if k > f:  # defensive: cannot reference before the scene started
            k = f

        if k > 0:
            # lidar(t-k) <- lidar(t)
            T_old_from_now = self.ego_pose_inv[f - k] @ self.ego_pose[f]
            # lidar(t) <- lidar(t-k)
            self._honest_T = self.ego_pose_inv[f] @ self.ego_pose[f - k]
            self._honest_dt = k * NUSC_DT
            if self.mode in ("key", "both"):
                # Un-project stale pixels at their true capture pose, then
                # express the result in the current lidar frame.
                data["lidar2img"] = self.lidar2img[f - k] @ T_old_from_now
        else:
            self._honest_T = None
            self._honest_dt = 0.0

        self.frame_in_scene += 1
        return self._orig_forward(memory_center, img_metas, topk_indexes, **data)

    # -- query-side conditioning -------------------------------------------

    def _honest_temporal_alignment(self, query_pos, tgt, reference_points):
        """temporal_alignment with the true capture pose and age for new queries.

        Mirrors StreamPETRHead.temporal_alignment exactly, except that the
        newly-born queries are conditioned on (T, dt) rather than the hardcoded
        (identity, 0). The carried-memory path is untouched -- it was always
        given the truth.
        """
        from projects.mmdet3d_plugin.models.utils.positional_encoding import (
            nerf_positional_encoding, pos2posemb1d, pos2posemb3d)

        head = self.head
        B = query_pos.size(0)
        dt = self._honest_dt
        T = self._honest_T

        temp_reference_point = ((head.memory_reference_point - head.pc_range[:3])
                                / (head.pc_range[3:6] - head.pc_range[0:3]))
        temp_pos = head.query_embedding(pos2posemb3d(temp_reference_point))
        temp_memory = head.memory_embedding

        if T is None:
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0)\
                .unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        else:
            rec_ego_pose = T.to(query_pos.device).view(1, 1, 4, 4)\
                .repeat(B, query_pos.size(1), 1, 1).float()

        if head.with_ego_pos:
            # Slots [0:3] mirror the memory layout [velo_x, velo_y, timestamp];
            # the upstream code leaves all three at zero for new queries.
            rec_prefix = torch.zeros_like(reference_points[..., :3])
            rec_prefix[..., 2] = dt
            rec_ego_motion = torch.cat(
                [rec_prefix, rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = head.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = head.ego_pose_pe(query_pos, rec_ego_motion)

            memory_ego_motion = torch.cat(
                [head.memory_velo, head.memory_timestamp,
                 head.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = head.ego_pose_memory(temp_memory, memory_ego_motion)

        # The new queries' age: dt instead of the hardcoded zero.
        query_pos += head.time_embedding(
            pos2posemb1d(torch.full_like(reference_points[..., :1], dt)))
        temp_pos += head.time_embedding(pos2posemb1d(head.memory_timestamp).float())

        if head.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :head.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :head.num_propagated]],
                                  dim=1)
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, :head.num_propagated]],
                dim=1)
            n_out = query_pos.shape[1] + head.num_propagated
            if T is None:
                rec_ego_pose = torch.eye(4, device=query_pos.device)\
                    .unsqueeze(0).unsqueeze(0).repeat(B, n_out, 1, 1)
            else:
                # Stamped onto the new block by post_update_memory as
                # memory_egopose = ego_pose(t) @ T = ego_pose(t-k), i.e. the
                # pose where these photons were actually captured.
                rec_ego_pose = T.to(query_pos.device).view(1, 1, 4, 4)\
                    .repeat(B, n_out, 1, 1).float()
            temp_memory = temp_memory[:, head.num_propagated:]
            temp_pos = temp_pos[:, head.num_propagated:]

        return (tgt, query_pos, reference_points, temp_memory, temp_pos,
                rec_ego_pose)

    def _post_update_hook(self, data, rec_ego_pose, *args, **kwargs):
        """Age the new block by dt (rec_timestamp is hardcoded to 0 upstream)."""
        out = self._orig_post_update(data, rec_ego_pose, *args, **kwargs)
        if self._honest_dt:
            n_new = self.head.topk_proposals
            self.head.memory_timestamp[:, :n_new] += self._honest_dt
        return out


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="jsonfile_prefix for results")
    ap.add_argument("--delay-all", type=int, default=1,
                    help="uniform per-cam delay in frames (0 = fresh)")
    ap.add_argument("--delay-per-cam", default=None,
                    help="comma-separated 6 ints overriding --delay-all")
    ap.add_argument("--honest-mode", choices=HONEST_MODES, default="both",
                    help="which half of the honest protocol to apply")
    args = ap.parse_args()

    delays = _parse_delay(args)
    print(f"Per-cam delays (frames): {delays}")
    print(f"Honest mode: {args.honest_mode}")
    if args.honest_mode != "none" and len(set(delays)) > 1:
        # The new block is born from one cross-attention over all 6 cameras, so
        # a single (T, dt) cannot describe per-camera-varying staleness.
        raise SystemExit("honest modes require a uniform per-cam delay")

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    spg = cfg.data.test.pop("samples_per_gpu", 1)
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset = build_dataset(cfg.data.test)
    model.module.CLASSES = dataset.CLASSES

    scene_starts = _compute_scene_starts(dataset)
    print(f"Dataset: {len(dataset)} samples across {len(scene_starts)} scenes")
    substitutor = FeatureSubstitutor(model, scene_starts, delays)
    HonestStaleness(model, substitutor, args.honest_mode)

    allowed_sample_tokens = {info["token"] for info in dataset.data_infos}
    _install_partial_blob_eval_patch(allowed_sample_tokens)

    loader = build_dataloader(
        dataset, samples_per_gpu=spg, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    t0 = time.time()
    outputs = single_gpu_test(model, loader)
    print(f"\nInference took {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"):
        eval_kwargs.pop(key, None)
    eval_kwargs.update({"metric": ["bbox"], "jsonfile_prefix": args.out})

    print("\nEvaluating with nuScenes-devkit ...")
    result = dataset.evaluate(outputs, **eval_kwargs)
    print("\n=== nuScenes evaluation ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
