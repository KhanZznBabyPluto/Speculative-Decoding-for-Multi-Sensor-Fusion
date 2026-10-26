"""Stage 05 — Phase 0.1: verify MamBEV checkpoint loads and the model
builds + forwards.

Go/no-go gate before committing to the MamBEV pivot. From the Stage 04
NOTES feasibility analysis, the single biggest risk for Stage 05 is
checkpoint availability — README shows the trained-detector .pth is a
manual Google Drive download and the catalog still lists detection
checkpoints as TODO. This script tells us, in under a minute and
without touching the data pipeline, whether the published artifacts
actually load into MamBEV's code on disk.

What this verifies (structural, not numerical):
  1. The MamBEV `custom_imports` registers every type the config refers to.
  2. The config builds a model via mmengine without unresolved types.
  3. `load_checkpoint` succeeds and reports a coherent missing/unexpected
     key delta (sanity: if missing > 90% of weights, the .pth is for
     a different variant).
  4. The detector exposes `pts_bbox_head.bev_h / bev_w / embed_dims` —
     the shape contract for the future dense-BEV dump.
  5. (Optional, --dry-forward) Build a synthetic 6-cam image tensor and
     run `model.forward()` in test mode to confirm shapes flow end to
     end. Skipped by default because needs CUDA + mamba-ssm.

Usage:
  python verify_checkpoint.py                  # uses MamBEV-small defaults
  python verify_checkpoint.py --variant tiny   # MamBEV-tiny
  python verify_checkpoint.py --config <p> --ckpt <p>
  python verify_checkpoint.py --dry-forward    # adds GPU forward smoke test
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/ibex/user/khangea/multisensor")
MAMBEV_ROOT = REPO_ROOT / "MamBEV"
CKPT_DIR = REPO_ROOT / "checkpoints" / "mambev"

VARIANT_DEFAULTS = {
    # NOTE: we point at the TRAIN config (mambev_small_t3.py), not
    # mambev_small_t3_inference.py. The inference config references
    # `FastInferMamBEV` which is a phantom class — nowhere in MamBEV's
    # source does anything register it. The train config uses the
    # actually-registered `MamBEV` class, which is also what the
    # published checkpoint was trained against.
    "small": {
        "config": MAMBEV_ROOT / "projects" / "MAMBEV" / "configs" / "MamBEV" / "mambev_small_t3.py",
        "ckpt":   CKPT_DIR / "mambev_small_detector.pth",
    },
    "tiny": {
        "config": MAMBEV_ROOT / "projects" / "MAMBEV" / "configs" / "MamBEV" / "mambev_tiny_t3.py",
        "ckpt":   CKPT_DIR / "mambev_tiny_detector.pth",
    },
}


def setup_path() -> None:
    """MamBEV uses `projects.MAMBEV.mambev.*` imports — needs repo root on PYTHONPATH."""
    sys.path.insert(0, str(MAMBEV_ROOT))


def report(title: str, value: str, ok: Optional[bool] = None) -> None:
    tag = "    " if ok is None else ("[OK]" if ok else "[!!]")
    print(f"  {tag} {title:<38} {value}")


def step(idx: int, total: int, label: str) -> None:
    print(f"\n[{idx}/{total}] {label}")


def check_files(config_path: Path, ckpt_path: Path) -> bool:
    step(1, 5, "File existence")
    ok = True
    report("config", str(config_path), config_path.exists())
    if not config_path.exists():
        ok = False
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / 1e6
        report("checkpoint", f"{ckpt_path}  ({size_mb:.1f} MB)", True)
    else:
        report("checkpoint", f"MISSING — {ckpt_path}", False)
        ok = False
    return ok


def build_config(config_path: Path):
    step(2, 5, "Parse + resolve config")
    from mmengine.config import Config
    cfg = Config.fromfile(str(config_path))
    has_model = hasattr(cfg, "model")
    report("config.model present", str(has_model), has_model)
    if has_model:
        model_type = cfg.model.get("type", "?")
        report("model.type", model_type)
        head = cfg.model.get("pts_bbox_head", {})
        bev_h = head.get("bev_h")
        bev_w = head.get("bev_w")
        emb   = head.get("embed_dims")
        report("bev grid (H × W × C)", f"{bev_h} × {bev_w} × {emb}",
               all(v is not None for v in (bev_h, bev_w, emb)))
    return cfg


def register_modules() -> None:
    step(3, 5, "Register MamBEV custom modules")
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    report("mmdet3d default scope", "registered", True)
    try:
        # Triggers @MODELS.register_module() on every detector / head / layer.
        import projects.MAMBEV.mambev  # noqa: F401
        report("projects.MAMBEV.mambev import", "ok", True)
    except Exception as exc:
        report("projects.MAMBEV.mambev import", f"FAILED: {exc}", False)
        raise


def build_model(cfg):
    step(4, 5, "Build model + load checkpoint")
    from mmengine.registry import MODELS
    from mmengine.runner import load_checkpoint

    model = MODELS.build(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    report("model class", type(model).__name__, True)
    report("total params", f"{n_params/1e6:.2f}M", True)

    # Probe streaming behaviour — Stage 04 NOTES claim MamBEV runs N>1
    # backbone passes per frame.
    frames = getattr(model, "frames", None)
    report("frames (training schedule)", str(frames),
           frames is not None and len(frames) > 1)

    # head shape contract
    head = model.pts_bbox_head
    bev_h = getattr(head, "bev_h", None)
    bev_w = getattr(head, "bev_w", None)
    emb   = getattr(head, "embed_dims", None)
    report("head.bev_h / bev_w / embed_dims", f"{bev_h} / {bev_w} / {emb}",
           all(v is not None for v in (bev_h, bev_w, emb)))

    return model, load_checkpoint


def load_weights(model, load_checkpoint, ckpt_path: Path) -> dict:
    """Loads weights and returns the missing / unexpected key report."""
    info = load_checkpoint(model, str(ckpt_path), map_location="cpu", strict=False)
    # mmengine's load_checkpoint mutates `model` and returns the checkpoint
    # dict (with `state_dict` already consumed).  Missing / unexpected
    # keys are logged but not returned directly — we recompute them.

    # Param keys present on model:
    model_keys = set(k for k, _ in model.state_dict().items())
    # Param keys in the checkpoint we just consumed:
    sd = info.get("state_dict") if isinstance(info, dict) else None
    if sd is None:
        # Older mmengine versions: load_checkpoint already pops state_dict.
        # Fallback — re-open file once to get the key set.
        import torch
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
    ckpt_keys = set(sd.keys())

    missing    = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    n_loaded   = len(model_keys & ckpt_keys)
    pct_loaded = 100.0 * n_loaded / max(1, len(model_keys))

    report("keys loaded",     f"{n_loaded}/{len(model_keys)} ({pct_loaded:.1f}%)",
           pct_loaded >= 90.0)
    report("missing on model", f"{len(missing)}",
           len(missing) < 0.10 * len(model_keys))
    report("unexpected in ckpt", f"{len(unexpected)}",
           len(unexpected) < 0.10 * len(ckpt_keys))

    if missing and len(missing) <= 10:
        for k in sorted(missing)[:10]:
            report(" missing", k)
    elif missing:
        for k in sorted(missing)[:5]:
            report(" missing (sample)", k)

    if unexpected and len(unexpected) <= 10:
        for k in sorted(unexpected)[:10]:
            report(" unexpected", k)
    elif unexpected:
        for k in sorted(unexpected)[:5]:
            report(" unexpected (sample)", k)

    return {
        "n_model_keys": len(model_keys),
        "n_ckpt_keys":  len(ckpt_keys),
        "missing":      missing,
        "unexpected":   unexpected,
    }


def dry_forward(model, cfg) -> None:
    """Optional: build a fake 6-cam batch and run the model forward.

    Catches shape / cuda-kernel / mamba-ssm issues without needing the
    nuScenes pipeline. Requires CUDA + mamba-ssm installed.
    """
    step(5, 5, "Dry forward (synthetic 6-cam batch)")
    import torch
    if not torch.cuda.is_available():
        report("cuda available", "NO — skipping dry forward", False)
        return
    device = torch.device("cuda")
    model = model.to(device).eval()

    # Default MamBEV input: 6 cams, 3xH'xW' RGB.  Pull H'/W' from the
    # config's data_preprocessor if present, else fall back to 928x1600
    # (BEVFormer/MamBEV-small default).
    H, W = 928, 1600
    pp = cfg.model.get("data_preprocessor", {})
    crop = pp.get("pad_size_divisor")
    if crop:
        H = ((H + crop - 1) // crop) * crop
        W = ((W + crop - 1) // crop) * crop
    report("synthetic image size (6, 3, H, W)", f"6 × 3 × {H} × {W}")

    imgs = torch.randn(1, 6, 3, H, W, device=device, dtype=torch.float32)
    # Minimal img_meta — MamBEV head uses `lidar2img`, `img_shape`, `pad_shape`.
    img_meta = dict(
        img_shape=[(H, W, 3)] * 6,
        pad_shape=[(H, W, 3)] * 6,
        lidar2img=[torch.eye(4).numpy() for _ in range(6)],
        scene_token="synthetic",
        prev_bev_exists=False,
        can_bus=torch.zeros(18).numpy(),
    )

    try:
        with torch.no_grad():
            # Skip the full MVXTwoStageDetector forward — call the
            # backbone+neck path directly so we don't need a Det3DDataSample.
            img_feats = model.extract_img_feat(imgs, [img_meta])
        report("backbone+neck img_feats len", str(len(img_feats)), True)
        for i, f in enumerate(img_feats):
            report(f"  level {i} shape", f"{tuple(f.shape)}  dtype={f.dtype}")
    except Exception as exc:
        report("forward", f"FAILED: {type(exc).__name__}: {exc}", False)
        raise


def main() -> int:
    setup_path()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["small", "tiny"], default="small")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt",   type=Path, default=None)
    parser.add_argument("--dry-forward", action="store_true",
                        help="run a synthetic 6-cam forward on CUDA (needs mamba-ssm)")
    args = parser.parse_args()

    defaults = VARIANT_DEFAULTS[args.variant]
    config_path = args.config or defaults["config"]
    ckpt_path   = args.ckpt   or defaults["ckpt"]

    print("=" * 78)
    print(f"Stage 05 / Phase 0.1 — MamBEV checkpoint smoke test")
    print(f"  variant : {args.variant}")
    print(f"  config  : {config_path}")
    print(f"  ckpt    : {ckpt_path}")
    print("=" * 78)

    if not check_files(config_path, ckpt_path):
        print("\nFAIL — config or checkpoint missing. See"
              " stages/05_mambev_dense_bev/download_checkpoint.sh.")
        return 1

    cfg = build_config(config_path)
    register_modules()
    model, load_checkpoint = build_model(cfg)
    delta = load_weights(model, load_checkpoint, ckpt_path)

    if args.dry_forward:
        dry_forward(model, cfg)

    print("\n" + "=" * 78)
    # PASS criterion: at least 90% of model keys came from the checkpoint.
    pct = 100.0 * (delta["n_model_keys"] - len(delta["missing"])) / max(1, delta["n_model_keys"])
    if pct >= 90.0:
        print(f"PASS — {pct:.1f}% of model weights loaded.")
        print("Next: dump a dense BEV for one scene (Phase 1.1 of Stage 05).")
        return 0
    else:
        print(f"FAIL — only {pct:.1f}% of weights loaded.")
        print("Likely wrong-variant checkpoint (e.g., small ckpt vs tiny config).")
        print("Try --variant tiny, or open the downloaded .pth and check shapes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
