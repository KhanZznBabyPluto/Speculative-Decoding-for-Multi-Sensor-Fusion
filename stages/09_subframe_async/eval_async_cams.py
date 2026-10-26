"""
Evaluate StreamPETR when some or all cameras deliver OLDER 12 Hz sweep images.

Sub-keyframe staleness (~80-500 ms) and per-camera asynchrony, done purely on the
data side: before inference, each val sample's camera entries are rewritten to
point at an earlier sweep (from build_sweep_index.py). No model hooks — the whole
backbone runs on the actual older pixels, and memory evolves exactly as a real
async stack would feed it.

  --geometry naive   sweep image + the keyframe's extrinsics (the model is told
                     the photo is current: the synchronous-interface condition)
  --geometry honest  sweep image + its true capture extrinsics in lidar(t)
                     (the key-side honest lift, exact per camera)

Schedules (age in sweep steps, 1 step ≈ 80 ms; camera order = pkl order
FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT):
  uniform:J        every camera J steps old
  static:a0,..,a5  fixed per-camera ages
  rr:P             round-robin: 6/P cameras refreshed per tick; camera c has age
                   (phase - c) mod P  -> at any moment ages 0..P-1 are all present
  whole:P          all 6 cameras refreshed together every P ticks; all share age
                   phase mod P  (same compute as rr:P, same mean age)
  rand:J           every frame all 6 cameras share one age drawn U{0..J}, seeded
                   by the sample token (delay that jumps frame to frame)
The tick phase at a keyframe is arbitrary on real hardware, so phase =
frame_idx mod P cycles it across frames.

Run from inside StreamPETR/ (GPU, a100 — flash-attn):
    PYTHONPATH=$PWD python ../stages/09_subframe_async/eval_async_cams.py \
        --schedule uniform:1 --geometry honest \
        --out ../stages/09_subframe_async/dumps/async_uniform1_honest

Mode switching (stage 10d): `--switch-ckpt CKPT --switch-above-ms T` loads CKPT's
weights before every frame whose max realised camera age exceeds T ms and the
`--ckpt` weights otherwise. The head's temporal memory lives in plain attributes,
not the state_dict, so it is carried ACROSS switches — as in a real stack.
"""

import argparse
import copy
import json
import os
import pickle
import random
import sys
import time

import numpy as np

PROJECT_DIR = "/ibex/user/khangea/multisensor"
INDEX_PKL = os.path.join(PROJECT_DIR, "stages/09_subframe_async/dumps/sweep_index_val.pkl")
CONFIG = "projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py"
CKPT = os.path.join(PROJECT_DIR, "checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth")
N_CAMS = 6


# ── Schedules ────────────────────────────────────────────────────────────────

def parse_schedule(spec):
    """Turn a schedule string into a function (frame_idx, token) -> list of 6 ages (steps)."""
    kind, _, arg = spec.partition(":")
    if kind == "uniform":
        j = int(arg)
        return lambda f, tok: [j] * N_CAMS
    if kind == "static":
        ages = [int(a) for a in arg.split(",")]
        assert len(ages) == N_CAMS, spec
        return lambda f, tok: list(ages)
    if kind == "rr":
        p = int(arg)
        return lambda f, tok: [(f - c) % p for c in range(N_CAMS)]
    if kind == "whole":
        p = int(arg)
        return lambda f, tok: [f % p] * N_CAMS
    if kind == "rand":
        j_max = int(arg)
        return lambda f, tok: [random.Random(tok).randint(0, j_max)] * N_CAMS
    raise ValueError(f"unknown schedule {spec!r}")


def apply_schedule(infos, index, ages_of, geometry):
    """Rewrite camera entries of `infos` in place to point at older sweeps.

    Args:
        infos: dataset.data_infos (list of dicts, pkl layout).
        index: token -> cam -> list[entry|None] from build_sweep_index.py.
        ages_of: (frame_idx, token) -> list of 6 requested ages in sweep steps.
        geometry: 'naive' keeps the keyframe extrinsics; 'honest' uses the
            sweep's true capture extrinsics in lidar(t).

    Returns:
        dict token -> list of 6 realised ages in ms (a request past scene start
        clamps to the oldest sweep that exists, so realised can be < requested).
    """
    realised = {}
    for info in infos:
        requested = ages_of(info["frame_idx"], info["token"])
        row = []
        for c, (cam, cam_info) in enumerate(info["cams"].items()):
            chain = index[info["token"]][cam]
            j = requested[c]
            while chain[j] is None:           # scene start: clamp
                j -= 1
            if j == 0:                        # untouched, so uniform:0 is bit-identical to fresh
                row.append(0.0)
                continue
            e = chain[j]
            new = copy.copy(cam_info)
            new["data_path"] = e["data_path"]
            if geometry == "honest":
                new["sensor2lidar_rotation"] = e["sensor2lidar_rotation"]
                new["sensor2lidar_translation"] = e["sensor2lidar_translation"]
                new["timestamp"] = e["timestamp"]
            info["cams"][cam] = new
            row.append((chain[0]["timestamp"] - e["timestamp"]) / 1e3)
        realised[info["token"]] = row
    return realised


def summarise_ages(realised):
    """Mean / max realised age over all (sample, camera) pairs, in ms."""
    a = np.array(list(realised.values()))          # (n_samples, 6)
    return dict(mean_ms=float(a.mean()), max_ms=float(a.max()),
                frac_fresh=float((a == 0).mean()))


def install_mode_switch(detector, primary_sd, switch_sd, infos, use_switch):
    """Swap weights before each frame according to `use_switch[token]`.

    img_metas carry no sample token, but the test loader is sequential and
    unshuffled, so frame i is infos[i]; every frame is checked against its
    scene_token and first-camera filename, so a desync fails loudly.

    Args:
        detector: the Petr3D module (model.module).
        primary_sd / switch_sd: state dicts on the detector's device; only keys
            whose tensors differ are copied per switch.
        infos: dataset.data_infos AFTER apply_schedule (loader order).
        use_switch: token -> bool, True = run this frame with switch_sd.

    Returns:
        dict of counters (frames per mode, number of weight swaps).
    """
    import torch
    keys = [k for k in primary_sd if not torch.equal(primary_sd[k], switch_sd[k])]
    params = dict(detector.state_dict(keep_vars=True))
    stats = dict(primary=0, switch=0, swaps=0, differing_tensors=len(keys))
    state = {"mode": "primary", "i": 0}
    orig = detector.simple_test

    def hook(img_metas, **data):
        info = infos[state["i"]]
        state["i"] += 1
        meta = img_metas[0]
        first_cam = next(iter(info["cams"].values()))["data_path"]
        assert meta["scene_token"] == info["scene_token"] and \
            os.path.basename(meta["filename"][0]) == os.path.basename(first_cam), \
            f"loader/infos desync at frame {state['i'] - 1}"
        mode = "switch" if use_switch[info["token"]] else "primary"
        if mode != state["mode"]:
            src = switch_sd if mode == "switch" else primary_sd
            with torch.no_grad():
                for k in keys:
                    params[k].data.copy_(src[k])
            state["mode"] = mode
            stats["swaps"] += 1
        stats[mode] += 1
        return orig(img_metas, **data)

    detector.simple_test = hook
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="StreamPETR eval with stale 12 Hz camera sweeps")
    ap.add_argument("--schedule", default="uniform:0")
    ap.add_argument("--geometry", choices=("naive", "honest"), default="honest")
    ap.add_argument("--out", required=True, help="jsonfile_prefix for result files")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--index", default=INDEX_PKL)
    ap.add_argument("--switch-ckpt", default=None,
                    help="weights for frames whose max camera age > --switch-above-ms")
    ap.add_argument("--switch-above-ms", type=float, default=0.0)
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite")

    import torch  # noqa: F401  (heavy imports only on the GPU path)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    import projects.mmdet3d_plugin  # noqa: F401
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    sys.path.insert(0, os.path.join(PROJECT_DIR, "stages/04_streampetr_latent"))
    from measure_stale_map import _install_partial_blob_eval_patch

    cfg = Config.fromfile(args.config)
    if cfg.get("custom_imports"):          # e.g. stage-10 age-aware head; module dir must be on PYTHONPATH
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.data.test.test_mode = True
    spg = cfg.data.test.pop("samples_per_gpu", 1)
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset = build_dataset(cfg.data.test)
    model.module.CLASSES = dataset.CLASSES

    index = pickle.load(open(args.index, "rb"))["index"]
    realised = apply_schedule(dataset.data_infos, index,
                              parse_schedule(args.schedule), args.geometry)
    ages = summarise_ages(realised)
    print(f"schedule={args.schedule} geometry={args.geometry} "
          f"realised age mean={ages['mean_ms']:.1f} ms max={ages['max_ms']:.0f} ms "
          f"fresh share={ages['frac_fresh']:.2f}")

    switch_stats = None
    if args.switch_ckpt:
        import torch
        dev = next(model.module.parameters()).device
        primary_sd = {k: v.detach().clone() for k, v in model.module.state_dict().items()}
        raw = torch.load(args.switch_ckpt, map_location="cpu")["state_dict"]
        assert set(raw) == set(primary_sd), "switch checkpoint has a different architecture"
        switch_sd = {k: v.to(dev) for k, v in raw.items()}
        use_switch = {tok: max(a) > args.switch_above_ms for tok, a in realised.items()}
        switch_stats = install_mode_switch(model.module, primary_sd, switch_sd,
                                           dataset.data_infos, use_switch)
        print(f"mode switch: {args.switch_ckpt} when max age > {args.switch_above_ms} ms "
              f"({sum(use_switch.values())}/{len(use_switch)} frames; "
              f"{switch_stats['differing_tensors']} tensors differ)")

    _install_partial_blob_eval_patch({info["token"] for info in dataset.data_infos})
    loader = build_dataloader(dataset, samples_per_gpu=spg,
                              workers_per_gpu=cfg.data.workers_per_gpu,
                              dist=False, shuffle=False,
                              nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    t0 = time.time()
    outputs = single_gpu_test(model, loader)
    print(f"\nInference took {time.time() - t0:.1f}s")
    if switch_stats:
        print(f"mode switch stats: {switch_stats}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "realised_ages.json"), "w") as f:
        json.dump(dict(schedule=args.schedule, geometry=args.geometry,
                       summary=ages, per_token_ms=realised,
                       switch=dict(ckpt=args.switch_ckpt, above_ms=args.switch_above_ms,
                                   stats=switch_stats) if args.switch_ckpt else None), f)

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"):
        eval_kwargs.pop(key, None)
    eval_kwargs.update({"metric": ["bbox"], "jsonfile_prefix": args.out})
    result = dataset.evaluate(outputs, **eval_kwargs)
    print("\n=== nuScenes evaluation ===")
    print(f"  schedule={args.schedule}  geometry={args.geometry}  "
          f"mean age={ages['mean_ms']:.1f} ms")
    print(f"  mAP={result['pts_bbox_NuScenes/mAP']:.4f}  "
          f"NDS={result['pts_bbox_NuScenes/NDS']:.4f}")
    for k in ("mATE", "mASE", "mAOE", "mAVE", "mAAE"):
        key = f"pts_bbox_NuScenes/{k}"
        if key in result:
            print(f"  {k}={result[key]:.4f}")


if __name__ == "__main__":
    main()
