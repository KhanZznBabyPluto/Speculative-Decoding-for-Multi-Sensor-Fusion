"""Phase 3.3: speculator training dataset.

Enumerates `(src, tgt)` frame pairs from dumped scenes (schema_version
≥ 2) and yields per-pair training tuples. The dataset has two roles:

  (1) Build the prediction target = the actual NEW block at `tgt`
      (`memory_embedding[0:256]` and `memory_reference_point[0:256]`).
  (2) Build the speculator's *input* including the AsyncSchedule
      effect: each camera's features may come from an earlier frame
      than `tgt`, simulating arrival delay.

## Coordinate normalization in ego pose

We subtract `tgt.ego_pose[:3, 3]` from every refpoint (carried_ref,
src_new_ref, tgt_new_ref) so they become centered around the target
ego. All speculator inputs and the training target live in this
*ego-centered* frame. Metric reporting still happens in metres because
centering is a pure translation; L2 norms are unchanged.

## Per-cam staleness sampling

For each pair, we sample a staleness vector`s ∈ [0, k]^N_cams`. 
The features for camera c are pulled from `frames[tgt - s[c]].img_feats[c]`.
Setting `s = 0` everywhere recovers the "all fresh" case
(= what StreamPETR's head would have seen).
Setting `s = k` everywhere is "fully stale" (= use src's
features).

The smoke-test training uses k=1, s=0 only — the cleanest possible
overfit setting. Random-schedule training is exposed as a flag for
Phase 3.4 onward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_PROPAGATED = 256
MEMORY_LEN = 1024

# Refpoint scaling. Ego-centered global coords are ~±100 m; the speculator's
# ref-PE MLP and its ref-delta output train more stably on ~unit-scale inputs.
# We divide all refs (carried, src_new, tgt_new) by this factor; the training
# loop multiplies refpoint metrics back to metres for reporting.
REF_SCALE = 50.0


@dataclass
class DatasetConfig:
    dump_paths: Tuple[str, ...]
    k_min: int = 1
    k_max: int = 1
    schedule_mode: str = "fresh"     # "fresh" | "random"
    max_staleness: int = 1           # only used when schedule_mode == "random"
    seed: int = 0
    # If set, only scenes whose token is in this set contribute pairs.
    # Used by make_train_val_split to build disjoint train/val datasets.
    scene_filter: Optional[Set[str]] = None


class SpeculatorDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.scenes: List[List[dict]] = []
        self.scene_tokens: List[str] = []
        for p in cfg.dump_paths:
            dump = torch.load(p, map_location="cpu", weights_only=False)
            for scene_tok, scene in dump.items():
                if scene.get("schema_version", 1) < 2:
                    raise ValueError(
                        f"{p}: scene {scene_tok} has schema_version "
                        f"{scene.get('schema_version')} (< 2). Re-dump with "
                        f"dump_latents.py at the current version."
                    )
                self.scenes.append(scene["frames"])
                self.scene_tokens.append(scene_tok)

        self.pairs: List[Tuple[int, int, int]] = []
        for s_idx, frames in enumerate(self.scenes):
            if cfg.scene_filter is not None:
                if self.scene_tokens[s_idx] not in cfg.scene_filter:
                    continue
            T = len(frames)
            for src in range(T):
                for k in range(cfg.k_min, cfg.k_max + 1):
                    tgt = src + k
                    if tgt < T:
                        self.pairs.append((s_idx, src, tgt))

        self._rng = np.random.default_rng(cfg.seed)

    @property
    def active_scene_tokens(self) -> List[str]:
        """Scene tokens that actually contribute pairs to this dataset."""
        if self.cfg.scene_filter is None:
            return list(self.scene_tokens)
        return [t for t in self.scene_tokens if t in self.cfg.scene_filter]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s_idx, src_i, tgt_i = self.pairs[idx]
        frames = self.scenes[s_idx]
        src_f = frames[src_i]
        tgt_f = frames[tgt_i]
        k = tgt_i - src_i

        # --- staleness per cam --------------------------------------------
        n_cams = src_f["img_feats"].shape[0]
        if self.cfg.schedule_mode == "fresh":
            staleness = np.zeros(n_cams, dtype=np.int64)
        elif self.cfg.schedule_mode == "random":
            hi = min(k, self.cfg.max_staleness)
            staleness = self._rng.integers(0, hi + 1, size=n_cams)
        else:
            raise ValueError(self.cfg.schedule_mode)

        # --- pull per-cam image features (each from frame[tgt - s[c]]) ---
        per_cam_feats = []
        for c in range(n_cams):
            f_idx = tgt_i - int(staleness[c])
            per_cam_feats.append(frames[f_idx]["img_feats"][c])
        img_feats = torch.stack(per_cam_feats, dim=0)              # [N, C, H, W]

        # --- ego-center + scale reference points -------------------------
        center = torch.from_numpy(tgt_f["ego_pose"][:3, 3]).float()  # [3]

        def centered_scaled(ref):
            return (ref.float() - center) / REF_SCALE

        carried_emb = src_f["memory_embedding"][:MEMORY_LEN].float()
        carried_ref = centered_scaled(src_f["memory_reference_point"][:MEMORY_LEN])
        src_new_emb = src_f["memory_embedding"][:NUM_PROPAGATED].float()
        src_new_ref = centered_scaled(src_f["memory_reference_point"][:NUM_PROPAGATED])
        tgt_new_emb = tgt_f["memory_embedding"][:NUM_PROPAGATED].float()
        tgt_new_ref = centered_scaled(tgt_f["memory_reference_point"][:NUM_PROPAGATED])

        return {
            "carried_emb":  carried_emb,
            "carried_ref":  carried_ref,
            "img_feats":    img_feats,
            "staleness":    torch.from_numpy(staleness).long(),
            "src_new_emb":  src_new_emb,
            "src_new_ref":  src_new_ref,
            "tgt_new_emb":  tgt_new_emb,
            "tgt_new_ref":  tgt_new_ref,
            "k":            k,
        }


def collate(batch):
    """Default-ish collate: stack everything along batch dim."""
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out


def _peek_scene_tokens(dump_paths: Tuple[str, ...]) -> List[str]:
    """Read dump headers and return scene tokens in the order they appear.
    Has to torch.load the full dump (no header-only mode in pt), so adds
    ~5s/GB of startup time. Acceptable for current dump sizes."""
    tokens: List[str] = []
    for p in dump_paths:
        dump = torch.load(p, map_location="cpu", weights_only=False)
        tokens.extend(dump.keys())
        del dump
    return tokens


def make_train_val_split(
    dump_paths: Tuple[str, ...],
    n_val_scenes: int,
    k_min: int = 1, k_max: int = 1,
    schedule_mode: str = "fresh",
    max_staleness: int = 1,
    seed: int = 0,
) -> Tuple["SpeculatorDataset", "SpeculatorDataset"]:
    """Build a disjoint scene-level train/val split.

    The last `n_val_scenes` scenes (in load order) are held out for
    validation; the rest go to training. Splitting BY SCENE TOKEN
    (not by frame-pair) is what makes the val set actually unseen —
    splitting by pair would let train and val share scene context
    through neighbouring frames, which is a soft form of leakage.

    Both datasets get the same k_min/k_max/schedule_mode so the metric
    is measured under the same conditions as training.
    """
    all_tokens = _peek_scene_tokens(dump_paths)
    if n_val_scenes <= 0 or n_val_scenes >= len(all_tokens):
        raise ValueError(
            f"n_val_scenes={n_val_scenes} must be in (0, {len(all_tokens)}); "
            f"got {len(all_tokens)} total scenes")
    val_tokens = set(all_tokens[-n_val_scenes:])
    train_tokens = set(all_tokens[:-n_val_scenes])

    common = dict(
        dump_paths=dump_paths,
        k_min=k_min, k_max=k_max,
        schedule_mode=schedule_mode,
        max_staleness=max_staleness,
        seed=seed,
    )
    train_ds = SpeculatorDataset(DatasetConfig(scene_filter=train_tokens, **common))
    val_ds   = SpeculatorDataset(DatasetConfig(scene_filter=val_tokens, **common))
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Smoke test — load scene_0.pt, build dataset, check one batch.
# ---------------------------------------------------------------------------
def _self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    dump_path = os.path.join(here, "dumps", "scene_0.pt")
    if not os.path.exists(dump_path):
        print(f"[SKIP] {dump_path} not present — re-run dump_latents.py first.")
        return

    cfg = DatasetConfig(
        dump_paths=(dump_path,),
        k_min=1, k_max=1, schedule_mode="fresh",
    )
    ds = SpeculatorDataset(cfg)
    print(f"[OK] dataset built: {len(ds)} pairs from {len(ds.scenes)} scene(s)")
    item = ds[0]
    expected_shapes = {
        "carried_emb":  (MEMORY_LEN, 256),
        "carried_ref":  (MEMORY_LEN, 3),
        "img_feats":    (6, 256, 16, 44),
        "staleness":    (6,),
        "src_new_emb":  (NUM_PROPAGATED, 256),
        "src_new_ref":  (NUM_PROPAGATED, 3),
        "tgt_new_emb":  (NUM_PROPAGATED, 256),
        "tgt_new_ref":  (NUM_PROPAGATED, 3),
    }
    for key, shape in expected_shapes.items():
        got = tuple(item[key].shape)
        assert got == shape, f"{key}: got {got}, expected {shape}"
    print("[OK] per-item shapes match expected")

    # Centering+scaling sanity: tgt_new_ref should be roughly in [-2, +2]
    # (= ±100 m / 50 m REF_SCALE).
    max_xy_scaled = item["tgt_new_ref"][:, :2].abs().max().item()
    print(f"  centered+scaled tgt_new_ref max |xy| = {max_xy_scaled:.2f} "
          f"(= {max_xy_scaled * REF_SCALE:.1f} m at REF_SCALE={REF_SCALE})")

    # Fresh schedule → staleness all zeros
    assert (item["staleness"] == 0).all(), item["staleness"]
    print("[OK] schedule_mode='fresh' produces zero staleness")

    # Batch collate
    batch = collate([ds[0], ds[1]])
    assert batch["img_feats"].shape == (2, 6, 256, 16, 44), batch["img_feats"].shape
    print("[OK] collate stacks batch dim correctly")

    print("\nAll dataset self-tests pass.")


if __name__ == "__main__":
    _self_test()
