"""Phase 3.4 diagnostic: measure StreamPETR's mAP degradation under
uniformly stale camera inputs.

The question we're answering:
    Does feeding StreamPETR stale camera features ACTUALLY hurt mAP?

We have never measured this. All prior Phase 2/3 work simulated staleness
only in the speculator's input space (substituting img_feats inside a
detached pipeline). Here we run the FULL StreamPETR head with stale
features and evaluate against nuScenes ground truth.

Mechanism: monkey-patch `Petr3D.extract_img_feat` to maintain a per-cam
ring buffer of recent feature tensors. After fresh features are computed
at frame t, we overwrite per-cam slots with `cache[c][t - delay_c]`
before returning. The head receives stale features for the configured
delay and produces detections normally.

Cache reset happens at scene boundaries (StreamPETR's `prev_scene_token`
mechanism handles memory reset analogously). At the first frame of any
scene the cache is empty, so the effective delay is clamped to 0 for
that frame regardless of the configured value.

This script reuses the patched `single_gpu_test` path (Phase 0.1) and
`dataset.evaluate` (nuScenes-devkit mAP/NDS). The only addition is the
feature substitution hook.

Run from inside StreamPETR/:

    cd /ibex/user/khangea/multisensor/StreamPETR
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    PYTHONPATH=$PWD python ../stages/04_streampetr_latent/measure_stale_map.py \
        --config projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
        --ckpt   ../checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth \
        --delay-all 0 \
        --out    ../stages/04_streampetr_latent/dumps/stale_map_delay0
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
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


N_CAMS = 6


class FeatureSubstitutor:
    """Monkey-patches `Petr3D.extract_img_feat` so the head sees stale
    per-cam features instead of the freshly computed ones.

    The cache stores past-frame per-cam features (post-FPN, the exact
    tensor the head consumes — same contract as Phase 3.1 dumps). At
    every frame, after computing the fresh features:
      1) we append per-cam slices to the per-cam cache;
      2) we overwrite each cam's slot in the output with
         `cache[c][frame_in_scene - delay_c]`, clamped at 0 so the
         first `delay_c` frames of each scene fall back to "fresh"
         (cache too short to satisfy the delay).

    Cache resets at scene boundaries. Scene boundaries are precomputed
    from `dataset.data_infos` in the same order the dataloader iterates,
    and we increment a sample counter inside the hook to detect them.
    """

    def __init__(self, model, scene_start_indices, delay_per_cam):
        self.detector = model.module
        self.delay_per_cam = list(delay_per_cam)
        self.scene_starts = set(scene_start_indices)
        self.cache = [[] for _ in range(N_CAMS)]
        self.frame_in_scene = 0
        self.sample_counter = 0
        # True iff this hook just substituted at least one cam's features
        # (= the current frame actually saw stale input). Used by
        # downstream mitigation hooks to know when to apply the override.
        self.was_stale = False
        # Worst-case actual delay applied this frame (max over cams,
        # AFTER cache-length clamping). vel_warp mitigation uses this to
        # know by how many frames to forward-project ref points.
        self.last_applied_delay = 0
        # When False, the hook is a passthrough: no caching, no
        # substitution, no counter increment. Used by the Phase 4
        # context-matched dump to run a SECOND head forward (with raw
        # fresh features) within the same loader iteration as the
        # stale forward, without polluting the substitutor's internal
        # state. Toggle to False before the fresh pass, back to True
        # before continuing the stale trajectory.
        self.enabled = True
        self._orig = self.detector.extract_img_feat
        self.detector.extract_img_feat = self._hook

    def _hook(self, *args, **kwargs):
        fresh = self._orig(*args, **kwargs)         # [B=1, N, C, H, W]
        if not self.enabled:
            # Passthrough: do not cache, do not substitute, do not
            # increment counters. Caller (e.g., context-matched dump)
            # is running a side forward that must see raw fresh.
            self.was_stale = False
            self.last_applied_delay = 0
            return fresh
        if self.sample_counter in self.scene_starts:
            self.cache = [[] for _ in range(N_CAMS)]
            self.frame_in_scene = 0

        # Append fresh per-cam to cache (clone to detach from view)
        for c in range(N_CAMS):
            self.cache[c].append(fresh[0, c].detach().clone())

        # Substitute: cam c reads cache[c][frame_in_scene - delay_c],
        # clamped to ≥ 0 (i.e., fall back to fresh when cache too short).
        out = fresh.clone()
        max_d_applied = 0
        for c in range(N_CAMS):
            d = min(self.delay_per_cam[c], self.frame_in_scene)
            if d > 0:
                idx = self.frame_in_scene - d
                out[0, c] = self.cache[c][idx]
                if d > max_d_applied:
                    max_d_applied = d
        self.last_applied_delay = max_d_applied
        self.was_stale = max_d_applied > 0

        self.sample_counter += 1
        self.frame_in_scene += 1
        return out


def _compute_scene_starts(dataset):
    starts = []
    prev = None
    for i, info in enumerate(dataset.data_infos):
        tok = info.get("scene_token") or info.get("scene_name")
        if tok != prev:
            starts.append(i)
            prev = tok
    return starts


def _install_partial_blob_eval_patch(allowed_sample_tokens):
    """Monkey-patch nuScenes-devkit's `load_gt` so it only returns
    annotations for samples in `allowed_sample_tokens`.

    Why: our val pkl covers ~23 scenes out of nuScenes-val's 150 (partial
    blob on the cluster). `NuScenesEval.__init__` asserts
    `set(pred.sample_tokens) == set(gt.sample_tokens)`. Without this patch
    the assertion fails because gt is loaded for the full 150-scene split.

    Filtering gt (not pred) is correct semantically: we want
    devkit-quality mAP/NDS computed on the subset we actually predicted.
    """
    import nuscenes.eval.common.loaders as loaders_mod
    import nuscenes.eval.detection.evaluate as eval_mod
    _orig_load_gt = loaders_mod.load_gt

    def patched_load_gt(nusc, eval_split, box_cls, verbose=False):
        all_boxes = _orig_load_gt(nusc, eval_split, box_cls, verbose=verbose)
        before = len(all_boxes.boxes)
        all_boxes.boxes = {tok: bs for tok, bs in all_boxes.boxes.items()
                           if tok in allowed_sample_tokens}
        if verbose:
            print(f"  [partial-blob patch] GT filtered "
                  f"{before} → {len(all_boxes.boxes)} sample tokens")
        return all_boxes

    loaders_mod.load_gt = patched_load_gt
    # The evaluate module imports load_gt at top level — patch the alias too
    if hasattr(eval_mod, "load_gt"):
        eval_mod.load_gt = patched_load_gt


def _parse_delay(args):
    if args.delay_per_cam is not None:
        delays = [int(x) for x in args.delay_per_cam.split(",")]
        if len(delays) != N_CAMS:
            raise ValueError(f"--delay-per-cam must have {N_CAMS} comma-separated ints")
        return delays
    return [args.delay_all] * N_CAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="jsonfile_prefix for result files")
    ap.add_argument("--delay-all", type=int, default=0,
                    help="uniform per-cam delay in frame units (0 = fresh)")
    ap.add_argument("--delay-per-cam", default=None,
                    help="comma-separated 6 ints overriding --delay-all")
    args = ap.parse_args()

    delays = _parse_delay(args)
    print(f"Per-cam delays (frames): {delays}")

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
    FeatureSubstitutor(model, scene_starts, delays)

    # Partial-blob workaround for nuScenes-devkit's strict val-split check.
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
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
