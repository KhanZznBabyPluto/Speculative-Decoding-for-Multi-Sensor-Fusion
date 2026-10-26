"""Phase 3.4 mitigation experiment: rewrite StreamPETR's new memory
block on stale frames and measure how that changes mAP.

Four modes are tested, each progressively informed by the previous
result. Each modifies `head.memory_embedding[:, :256, :]` (and possibly
`memory_reference_point`) AFTER `post_update_memory` completes — so the
head's internal state machine is untouched and only the persistent
memory state visible to the *next* forward call is modified.

  * `none`     — control. Same as `measure_stale_map.py` baseline.
  * `zero`     — replace the new-block embedding with zeros. Empirically
                 made things WORSE on every (delay, metric) combination.
                 Suggests that even "wrong" new tokens carry useful
                 information the carried block depends on.
  * `freeze`   — replace with the new block from the most recent FRESH
                 frame in this scene. Also empirically worse than
                 `none`, though less so. Same conclusion as `zero`.
  * `vel_warp` — preserve the head's own stale-derived new block (keep
                 its embedding, its semantic content), but forward-
                 project `memory_reference_point[:, :, :2]` by
                 `memory_velo × dt × k`, where `k` is the actual
                 staleness applied this frame. This is *temporal
                 calibration without re-running the head* — the head
                 produced detections for time `t-k`, and we re-aim them
                 to time `t` using its own learned velocity estimates.

Empirical motivation for `vel_warp`: the `zero` / `freeze` failures
showed that the stale-derived new tokens are NOT garbage — they are
spatially lagged detections. Discarding them removed signal the carried
block needs to roll forward (frame-by-frame propagation). Correcting
their position (rather than removing them) is the lighter-touch
intervention consistent with this finding.

Run from inside StreamPETR/:

    cd /ibex/user/khangea/multisensor/StreamPETR
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    PYTHONPATH=$PWD python ../stages/04_streampetr_latent/measure_stale_map_mitigation.py \
        --config projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
        --ckpt   ../checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth \
        --delay-all 1 \
        --mitigation zero \
        --out    ../stages/04_streampetr_latent/dumps/stale_map_d1_zero
"""

import argparse
import os
import sys
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Reuse the diagnostic's hooks + patch — single source of truth so the
# semantics of "stale" stay identical across the two scripts.
from measure_stale_map import (  # noqa: E402
    N_CAMS,
    FeatureSubstitutor,
    _compute_scene_starts,
    _install_partial_blob_eval_patch,
    _parse_delay,
)


NUM_PROPAGATED = 256
MEMORY_LEN = 1024
NUSC_DT_SECONDS = 0.5     # nuScenes keyframe rate (2 Hz)


def velo_lidar_to_global(velo, ego_pose):
    """Rotate head velocities from the lidar(t) frame into the global frame.

    After `post_update_memory`, `memory_reference_point` is in the GLOBAL frame
    (streampetr_head.py:372) but `memory_velo` is the raw regression output,
    which lives in lidar(t) (the converter rotates GT velocity global -> lidar).
    Adding one to the other without this rotation shifts ref points in an
    arbitrary direction (lidar->global yaw has median ~55 deg on our val set).

    Args:
        velo: [B, N, 2] velocities in lidar(t), m/s.
        ego_pose: [B, 4, 4] lidar(t) -> global transform (`data['ego_pose']`).
    Returns:
        [B, N, 2] velocities in the global frame.
    """
    rot = ego_pose[:, :3, :3].to(velo.dtype)                          # [B, 3, 3]
    velo3 = torch.cat([velo, torch.zeros_like(velo[..., :1])], dim=-1)  # [B, N, 3]
    return torch.einsum("bij,bnj->bni", rot, velo3)[..., :2]


class MemoryMitigator:
    """After every `simple_test` call, override the head's new memory
    block based on whether the most recent forward used stale features.

    Hooks `Petr3D.simple_test` (the outermost test-time entrypoint) and
    runs *after* it completes so the head has finished its
    `post_update_memory` step. At that point
    `head.memory_embedding[:, :NUM_PROPAGATED, :]` contains the freshly
    prepended new block in global frame — exactly the slot we want to
    rewrite.

    On a fresh frame (substitutor.was_stale == False), we cache the
    current new block as `last_fresh_new_block` for any subsequent
    freeze-mode override. On a stale frame, we apply the configured
    mitigation. Scene resets clear the cache (detected via the
    substitutor's frame_in_scene counter reset).
    """

    def __init__(self, model, substitutor: FeatureSubstitutor, mode: str,
                 vel_threshold: float = 1.0, ltc_module=None):
        assert mode in ("none", "zero", "freeze", "vel_warp",
                        "vel_warp_gated", "ltc", "ltc_ref_only",
                        "ltc_consistent"), mode
        if mode in ("ltc", "ltc_ref_only", "ltc_consistent"):
            assert ltc_module is not None, \
                f"--mitigation {mode} requires --ltc-ckpt"
        if mode == "ltc_consistent":
            assert getattr(ltc_module.cfg, "predict_velo", False), \
                "--mitigation ltc_consistent requires an LTC trained with " \
                "--predict-velo (so pred_velo is meaningful)"
        self.mode = mode
        self.vel_threshold = vel_threshold     # used only in vel_warp_gated
        self.ltc = ltc_module                  # used only in mode == "ltc"
        self.detector = model.module
        self.head = self.detector.pts_bbox_head
        self.substitutor = substitutor
        self.last_fresh_new_block = None       # [B, NUM_PROPAGATED, D]
        self._prev_frame_in_scene = 0          # detect scene resets
        self._orig = self.detector.simple_test
        self.detector.simple_test = self._hook

    def _hook(self, *args, **kwargs):
        result = self._orig(*args, **kwargs)
        # Skip if no mitigation requested at all.
        if self.mode == "none":
            return result

        # The substitutor's frame_in_scene was incremented inside
        # extract_img_feat. After the entire forward of this sample, it
        # reads as "(scene-local index of just-processed frame) + 1".
        # A reset to 1 here means we just processed frame 0 of a new
        # scene (which is always fresh due to cache clamping).
        if self.substitutor.frame_in_scene == 1:
            self.last_fresh_new_block = None

        if not self.substitutor.was_stale:
            # Fresh frame: cache the head's new block for future freezes.
            with torch.no_grad():
                self.last_fresh_new_block = (
                    self.head.memory_embedding[:, :NUM_PROPAGATED, :].detach().clone()
                )
            return result

        # Stale frame: rewrite the just-prepended new block.
        with torch.no_grad():
            new_emb = self.head.memory_embedding[:, :NUM_PROPAGATED, :]
            if self.mode == "zero":
                new_emb.zero_()
            elif self.mode == "freeze":
                if self.last_fresh_new_block is not None:
                    new_emb.copy_(self.last_fresh_new_block)
                else:
                    # Should not happen — frame 0 of every scene is fresh
                    # and caches last_fresh_new_block before any stale
                    # frame arrives. Fall back to zero defensively.
                    new_emb.zero_()
            elif self.mode == "vel_warp":
                # Keep the head's stale-derived embedding. Forward-project
                # its xy ref points by (memory_velo × dt × k). z is left
                # untouched (cars don't drive vertically at nuScenes
                # scales). k is the actual staleness applied this frame
                # by the substitutor (clamped against cache length, so
                # at scene start it may be < configured delay).
                # Before the frame fix, the lidar-frame velo was added to the
                # global-frame ref directly — dumps named `*vel_warp` (without
                # `_fixframe`) were produced by that buggy version.
                k = self.substitutor.last_applied_delay
                if k > 0:
                    new_ref = self.head.memory_reference_point[:, :NUM_PROPAGATED, :]
                    new_velo = velo_lidar_to_global(
                        self.head.memory_velo[:, :NUM_PROPAGATED, :], kwargs["ego_pose"])
                    new_ref[:, :, :2] = new_ref[:, :, :2] + new_velo * (NUSC_DT_SECONDS * k)
            elif self.mode == "vel_warp_gated":
                # Same as vel_warp, but only apply to tokens whose velocity
                # magnitude is above `vel_threshold` m/s. Static objects
                # (traffic_cone, barrier, parked cars) get noisy velocity
                # estimates that DEGRADE their detections under vel_warp.
                # Gating by magnitude isolates the moving tokens that
                # benefit from forward projection.
                k = self.substitutor.last_applied_delay
                if k > 0:
                    new_ref = self.head.memory_reference_point[:, :NUM_PROPAGATED, :]
                    new_velo = velo_lidar_to_global(
                        self.head.memory_velo[:, :NUM_PROPAGATED, :], kwargs["ego_pose"])
                    mag = new_velo.norm(dim=-1, keepdim=True)            # [B, N, 1]
                    mask = (mag > self.vel_threshold).float()            # [B, N, 1]
                    shift = new_velo * (NUSC_DT_SECONDS * k) * mask      # [B, N, 2]
                    new_ref[:, :, :2] = new_ref[:, :, :2] + shift
            elif self.mode in ("ltc", "ltc_ref_only", "ltc_consistent"):
                # Learned residual correction. The LTC module wraps the
                # vel_warp baseline analytically and adds a learned delta
                # — see ltc.py. We feed it the SAME tensors the dump
                # script (`dump_ltc_pairs.py`) recorded at training time:
                # new block emb/ref/velo + carried block emb/ref.
                #
                # Variants:
                #   ltc            — write back (emb, ref). v2 default.
                #                    Damaged mAP empirically; downstream
                #                    head's MLN saw inconsistent
                #                    (corrected emb/ref, stale velo).
                #   ltc_ref_only   — diagnostic; write only ref. Also
                #                    damaged mAP, ruling out
                #                    embedding-mod as sole cause.
                #   ltc_consistent — v3. Write all three (emb, ref, velo)
                #                    so MLN inputs are jointly corrected.
                #                    Requires --predict-velo at train.
                k = self.substitutor.last_applied_delay
                if k > 0:
                    new_emb = self.head.memory_embedding[:, :NUM_PROPAGATED, :]
                    new_ref = self.head.memory_reference_point[:, :NUM_PROPAGATED, :]
                    new_velo = self.head.memory_velo[:, :NUM_PROPAGATED, :]
                    carr_emb = self.head.memory_embedding[
                        :, NUM_PROPAGATED:NUM_PROPAGATED + MEMORY_LEN, :]
                    carr_ref = self.head.memory_reference_point[
                        :, NUM_PROPAGATED:NUM_PROPAGATED + MEMORY_LEN, :]
                    # Center refs by carried-block xy median (same
                    # convention as the training dataset).
                    center_xy = carr_ref[0, :, :2].median(dim=0).values
                    center = torch.tensor(
                        [center_xy[0], center_xy[1], 0.0],
                        device=new_ref.device)
                    new_ref_c = new_ref - center
                    carr_ref_c = carr_ref - center
                    k_tensor = torch.tensor([k], device=new_ref.device,
                                            dtype=torch.long)
                    pred_emb, pred_ref_c, pred_velo = self.ltc(
                        stale_new_emb=new_emb,
                        stale_new_ref=new_ref_c,
                        stale_new_velo=new_velo,
                        carried_emb=carr_emb,
                        carried_ref=carr_ref_c,
                        k=k_tensor,
                    )
                    # Write back ref. Optionally write back emb (skipped
                    # in `ltc_ref_only` mode) and velo (only in
                    # `ltc_consistent`).
                    if self.mode != "ltc_ref_only":
                        new_emb.copy_(pred_emb)
                    new_ref.copy_(pred_ref_c + center)
                    if self.mode == "ltc_consistent":
                        new_velo.copy_(pred_velo)
        return result


class _AgeStateView:
    """Live read-through from HonestStaleness's per-frame state to the age port.

    HonestStaleness computes this frame's (T, dt) inside its `head.forward`
    wrapper; the port reads them later in `temporal_alignment`, which runs
    inside that same forward. A dict copied at install time would therefore be
    permanently stale — this view resolves on each access instead.
    """

    def __init__(self, honest):
        self._honest = honest

    def get(self, key, default=None):
        if key == "T":
            return self._honest._honest_T
        if key == "dt":
            return self._honest._honest_dt
        return default


def _bridge_age_state(model, honest):
    """Point the age port at HonestStaleness's live (T, dt)."""
    model._age_state = _AgeStateView(honest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="jsonfile_prefix for result files")
    ap.add_argument("--delay-all", type=int, default=1,
                    help="uniform per-cam delay in frame units")
    ap.add_argument("--delay-per-cam", default=None,
                    help="comma-separated 6 ints overriding --delay-all")
    ap.add_argument("--mitigation",
                    choices=("none", "zero", "freeze", "vel_warp",
                            "vel_warp_gated", "ltc", "ltc_ref_only",
                            "ltc_consistent"),
                    default="zero")
    ap.add_argument("--age-port", default=None,
                    help="checkpoint from stage 07 train_age_port.py. Routes "
                         "image-born queries through the TRAINED conditioning "
                         "port instead of the hardcoded (identity, dt=0). "
                         "Requires --honest-mode key; the comparison that "
                         "answers the experiment is against honest key alone "
                         "(0.1849 mAP at delay=1)")
    ap.add_argument("--honest-mode", default="none",
                    choices=("none", "key", "query", "both"),
                    help="declare the stale frame's true capture pose/age to "
                         "the head before mitigating. 'none' reproduces the "
                         "naive harness (the pre-2026-07-17 condition); 'key' "
                         "is the honest protocol whose recovery is real "
                         "(+0.052 mAP at delay=1, 23/23 scenes). Mitigations "
                         "measured under 'none' sit on a contaminated baseline")
    ap.add_argument("--vel-threshold", type=float, default=1.0,
                    help="m/s threshold for vel_warp_gated (below = static, "
                         "no projection; above = moving, project forward)")
    ap.add_argument("--ltc-ckpt", default=None,
                    help="trained LTC checkpoint (.pt) — required when "
                         "--mitigation ltc")
    args = ap.parse_args()

    delays = _parse_delay(args)
    print(f"Per-cam delays (frames): {delays}")
    print(f"Mitigation              : {args.mitigation}")

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

    # Order matters: feature substitution runs first (inside the model's
    # extract_img_feat), then the head's simple_test, then our mitigator
    # hook fires on simple_test's exit.
    substitutor = FeatureSubstitutor(model, scene_starts, delays)

    # Hook order is safe: HonestStaleness (key mode) wraps the HEAD's forward,
    # which runs inside simple_test, while MemoryMitigator wraps simple_test
    # itself. The lift is corrected before the head computes, the memory is
    # edited after it finishes — no overlap.
    if args.honest_mode != "none":
        if len(set(delays)) > 1:
            raise SystemExit("honest modes require a uniform per-cam delay")
        from honest_staleness import HonestStaleness
        honest = HonestStaleness(model, substitutor, args.honest_mode)
        print(f"Honest protocol: {args.honest_mode}")

        if args.age_port is not None:
            if args.honest_mode != "key":
                raise SystemExit("--age-port requires --honest-mode key: the "
                                 "port is trained on top of correct geometry")
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "07_delay_conditioning"))
            from age_port import AgeConditioningPort, install_age_port
            port = AgeConditioningPort(model.module.pts_bbox_head).cuda().eval()
            blob = torch.load(args.age_port, map_location="cpu")
            port.load_state_dict(blob["state_dict"])
            # HonestStaleness (key mode) already publishes the frame's true
            # (T, dt) on the model; the port reads the same state.
            model.module._age_state = {"T": None, "dt": 0.0}
            install_age_port(model.module, port)
            _bridge_age_state(model.module, honest)
            print(f"Loaded age port from {args.age_port} "
                  f"({port.n_params():,} params, trained {blob.get('steps')} steps)")

    ltc_module = None
    if args.mitigation in ("ltc", "ltc_ref_only", "ltc_consistent"):
        if args.ltc_ckpt is None:
            raise SystemExit(f"--mitigation {args.mitigation} requires --ltc-ckpt")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ltc import LTC, LTCConfig
        blob = torch.load(args.ltc_ckpt, map_location="cpu", weights_only=False)
        cfg_kwargs = blob.get("cfg", {})
        ltc_module = LTC(LTCConfig(**cfg_kwargs)).eval()
        ltc_module.load_state_dict(blob["state_dict"])
        ltc_module = ltc_module.to(torch.device("cuda"))
        print(f"Loaded LTC from {args.ltc_ckpt} "
              f"({sum(p.numel() for p in ltc_module.parameters())} params)")

    MemoryMitigator(model, substitutor, args.mitigation,
                    vel_threshold=args.vel_threshold,
                    ltc_module=ltc_module)

    allowed_sample_tokens = {info["token"] for info in dataset.data_infos}
    _install_partial_blob_eval_patch(allowed_sample_tokens)

    loader = build_dataloader(
        dataset,
        samples_per_gpu=spg,
        workers_per_gpu=cfg.data.workers_per_gpu,
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

    print(f"\nEvaluating with nuScenes-devkit... (this writes JSONs near {args.out})")
    result = dataset.evaluate(outputs, **eval_kwargs)
    print("\n=== nuScenes evaluation ===")
    print(f"  delay={delays[0]}  mitigation={args.mitigation}")
    print(f"  mAP={result['pts_bbox_NuScenes/mAP']:.4f}  "
          f"NDS={result['pts_bbox_NuScenes/NDS']:.4f}")
    for k in ("mATE", "mASE", "mAOE", "mAVE", "mAAE"):
        key = f"pts_bbox_NuScenes/{k}"
        if key in result:
            print(f"  {k}={result[key]:.4f}")


if __name__ == "__main__":
    main()
