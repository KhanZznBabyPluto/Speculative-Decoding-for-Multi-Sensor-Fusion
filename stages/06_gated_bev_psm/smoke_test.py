"""Phase 0.1 — MamBEV runnability check.

Goal: confirm that the local MamBEV install can
  (1) load a config + checkpoint,
  (2) run one nuScenes sample forward and produce a non-empty Det3DDataSample,
  (3) expose the dense BEV tensor we plan to dump in dump_bev.py.

What we verify before doing anything else:
  - BEV shape is (B, embed_dims, bev_h, bev_w) after reshape, or
    (B, L=bev_h*bev_w, embed_dims) if the head returns flattened.
    Either is fine for our pipeline — we just need to know which.
  - BEV dtype, mean, std, range (raw stats — Phase 3.1 on StreamPETR found
    raw FPN features had std≈46 / range±400 which is critical for any
    LayerNorm decision downstream; same caution applies here).
  - That a top detection comes back with non-degenerate score.

Run from inside MamBEV/ with mambev_env activated:

    cd /ibex/user/khangea/multisensor/MamBEV
    conda activate /ibex/user/khangea/conda-environments/mambev_env
    PYTHONPATH=$PWD python ../stages/06_gated_bev_psm/smoke_test.py \\
        --config projects/MAMBEV/configs/MamBEV/mambev_small_t3_inference.py \\
        --ckpt   <path-to-mambev-checkpoint>.pth
"""

import argparse
import sys

import torch

# NOTE: import order matters — MamBEV registers modules at import time via
# projects.MAMBEV.mambev side effects, so we import after argparse.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to MamBEV config (e.g. mambev_small_t3_inference.py)")
    p.add_argument("--ckpt", required=True,
                   help="Path to MamBEV checkpoint .pth")
    p.add_argument("--n-samples", type=int, default=1,
                   help="How many val samples to run before bailing.")
    return p.parse_args()


def build_model_and_dataset(config_path, ckpt_path):
    """Build MamBEV model + val dataset from a config; load checkpoint.

    MamBEV is on the new mmengine stack (mmcv 2.x / mmdet3d 1.x). The legacy
    `build_model(...)` + `load_checkpoint(...)` pattern from stage 04 won't
    map 1:1 — we use Config + Runner-style construction here.

    NOTE: The config's `custom_imports` lists every project package whose
    @register_module side-effects must fire before MODELS.build can resolve
    the type. We honor that list explicitly — hard-coding just
    `projects.MAMBEV.mambev` would miss PETR / DETR3D / BEVFormer / mmcv_dep
    helpers that some heads/transformers depend on.

    NOTE 2: `mambev_small_t3_inference.py` declares `type="FastInferMamBEV"`
    which is NOT defined anywhere in this fork — only `type="MamBEV"` is
    registered. Use the non-inference config (e.g. `mambev_small_t3.py`)
    or change the config's `type=` before invoking this.
    """
    from mmengine.config import Config
    from mmengine.registry import MODELS, DATASETS
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmengine.registry import init_default_scope

    cfg = Config.fromfile(config_path)
    # Honor `custom_imports` + `default_scope` from the config exactly like
    # Runner.from_cfg would.
    ci = cfg.get("custom_imports")
    if ci is not None:
        import_modules_from_strings(**ci)
    if cfg.get("default_scope"):
        init_default_scope(cfg.default_scope)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.eval().cuda()

    # Build the val (test) dataset; MamBEV cfg usually exposes cfg.val_dataloader
    # or cfg.test_dataloader. We follow the same convention as test.py.
    dl_cfg = cfg.get("test_dataloader") or cfg.get("val_dataloader")
    if dl_cfg is None:
        raise RuntimeError("Config has neither test_dataloader nor val_dataloader.")
    dataset = DATASETS.build(dl_cfg.dataset)
    return cfg, model, dataset


class BEVCapture:
    """Stash the head's CURRENT-FRAME BEV via a monkey-patch on `.forward`.

    Why a monkey-patch instead of `register_forward_hook`:
      MamBEV's `predict()` at MamBEV/detectors/mambev.py:492 calls
          self.pts_bbox_head.forward(...)
      directly, not `self.pts_bbox_head(...)`. The forward-hook machinery
      only fires through `nn.Module.__call__`, so a `register_forward_hook`
      on `pts_bbox_head` never sees the current-frame call. We must wrap
      `forward` itself.

    The history path (`obtain_history_bev` at mambev.py:254) does go through
    `__call__`, but it returns a bare BEV tensor (the `only_bev=True` branch).
    That's not what we want, so we ignore it.

    `forward(mode='training')` returns a dict with `bev_embed`; we keep it.
    """

    def __init__(self):
        self.bev = None
        self.n_dict_hits = 0
        self.n_other_hits = 0

    def install(self, head: torch.nn.Module):
        """Wrap head.forward. Idempotent — repeated calls reinstall."""
        if hasattr(head, "_bevcapture_orig_forward"):
            head.forward = head._bevcapture_orig_forward  # restore
        head._bevcapture_orig_forward = head.forward

        def wrapped(*args, **kwargs):
            out = head._bevcapture_orig_forward(*args, **kwargs)
            if isinstance(out, dict) and "bev_embed" in out:
                self.bev = out["bev_embed"].detach()
                self.n_dict_hits += 1
            else:
                self.n_other_hits += 1
            return out

        head.forward = wrapped

    def remove(self, head: torch.nn.Module):
        if hasattr(head, "_bevcapture_orig_forward"):
            head.forward = head._bevcapture_orig_forward
            del head._bevcapture_orig_forward


def main():
    args = parse_args()
    cfg, model, dataset = build_model_and_dataset(args.config, args.ckpt)
    print(f"Model: {type(model).__name__}")
    print(f"Dataset: {type(dataset).__name__}, len={len(dataset)}")

    cap = BEVCapture()
    cap.install(model.pts_bbox_head)

    with torch.no_grad():
        for i in range(args.n_samples):
            sample = dataset[i]
            # MamBEV's data_preprocessor expects the structured input the
            # dataloader produces. Easiest: build a one-shot loader and pull
            # batch through the model the same way Runner does. We avoid
            # reimplementing collate here.
            from mmengine.dataset import default_collate
            batch = default_collate([sample])
            data = model.data_preprocessor(batch, training=False)
            out = model._run_forward(data, mode="predict")

            print(f"\n--- sample {i} ---")
            print(f"prediction type: {type(out)}")
            if isinstance(out, list) and len(out) > 0:
                ds = out[0]
                if hasattr(ds, "pred_instances_3d"):
                    scores = ds.pred_instances_3d.scores_3d
                    top = scores[:5].tolist() if len(scores) else []
                    print(f"top-5 scores: {[f'{s:.3f}' for s in top]}")
            print(f"captured BEV type: {type(cap.bev)}  "
                  f"(dict hits={cap.n_dict_hits}, other hits={cap.n_other_hits})")
            if isinstance(cap.bev, torch.Tensor):
                t = cap.bev
                print(f"  shape  : {tuple(t.shape)}")
                print(f"  dtype  : {t.dtype}")
                print(f"  device : {t.device}")
                print(f"  mean   : {t.float().mean().item():+.4f}")
                print(f"  std    : {t.float().std().item():+.4f}")
                print(f"  min/max: {t.float().min().item():+.4f} / "
                      f"{t.float().max().item():+.4f}")
            else:
                print("  BEV was not a single Tensor — inspect cap.bev manually.")

    cap.remove(model.pts_bbox_head)


if __name__ == "__main__":
    main()
