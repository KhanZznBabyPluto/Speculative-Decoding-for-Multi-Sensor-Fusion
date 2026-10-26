"""Stage 05 — Phase 2.1 (part 2 of 2): warp + retention analysis on the BEV pair.

Consumes the dump produced by dump_bev_pair.py. Computes the inverse-ego
warp that maps BEV(t)'s grid into BEV(t+1)'s grid, then measures how much
the warped BEV(t) resembles the freshly-encoded BEV(t+1). Baselines:

  * unwarped     — BEV(t) compared directly to BEV(t+1) (no motion compensation)
  * warped       — BEV(t) → grid_sample with ego transform → compare to BEV(t+1)
  * zeros        — null reference
  * random       — N(0, std(BEV(t+1))) noise of matched magnitude

If `warped` is substantially closer to `BEV(t+1)` than `unwarped` is, the
dense-BEV speculation story has structural support (ego warp closes most
of the gap; the residual is what a learned speculator would address).

If `warped` is NO better than `unwarped`, ego-only warp doesn't capture the
inter-frame change — this would be the dense-BEV analog of the Phase 3-4
walls and would mean the MamBEV pivot doesn't escape the failure modes.

If `warped` is much WORSE than `unwarped`, our warp math is wrong; need to
re-check the BEV / LiDAR / ego frame conventions before trusting any
follow-up.

This script does NOT yet feed the warped BEV through the head (that's a
follow-up). The L2/cosine numbers here are the cheap-first signal.

BEV physical extent from config (point_cloud_range):
  x ∈ [-51.2, 51.2], y ∈ [-51.2, 51.2]  ⇒ 102.4 m × 102.4 m, 100×100 cells
  cell size = 1.024 m
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


BEV_H, BEV_W      = 100, 100
PC_RANGE_XY_MIN   = -51.2
PC_RANGE_XY_MAX   =  51.2
PC_RANGE_LEN      = PC_RANGE_XY_MAX - PC_RANGE_XY_MIN  # = 102.4


# --------------------------------------------------------------------------- #
#  Tensor shape helpers                                                       #
# --------------------------------------------------------------------------- #

def reshape_bev(bev_flat: torch.Tensor) -> torch.Tensor:
    """Captured BEV from the encoder hook is `(groups, H*W, C)`. Reshape to
    `(groups, C, H, W)` for grid_sample."""
    assert bev_flat.dim() == 3, f"expected (groups, H*W, C), got {tuple(bev_flat.shape)}"
    groups, n_tokens, C = bev_flat.shape
    assert n_tokens == BEV_H * BEV_W, f"unexpected token count {n_tokens}"
    bev = bev_flat.view(groups, BEV_H, BEV_W, C)
    bev = bev.permute(0, 3, 1, 2).contiguous()  # (groups, C, H, W)
    return bev


# --------------------------------------------------------------------------- #
#  Warp math                                                                  #
# --------------------------------------------------------------------------- #

def lidar2global(ego2global: np.ndarray, lidar2ego: np.ndarray) -> np.ndarray:
    return ego2global @ lidar2ego


def compute_warp_T_l_t_from_l_t1(eg_t: np.ndarray, l2e_t: np.ndarray,
                                 eg_t1: np.ndarray, l2e_t1: np.ndarray) -> np.ndarray:
    """Returns T such that a point P expressed in LiDAR-frame-of-t+1 maps to
    LiDAR-frame-of-t by P_t = T @ P_t1.  We need this direction because for
    every output cell in BEV(t+1)'s coords we sample BEV(t) at the warped
    location.
    """
    l2g_t  = lidar2global(eg_t,  l2e_t)
    l2g_t1 = lidar2global(eg_t1, l2e_t1)
    return np.linalg.inv(l2g_t) @ l2g_t1


def build_grid_sample_grid(T_l_t_from_l_t1: np.ndarray, device,
                           convention: str = "cols=x,rows=y") -> torch.Tensor:
    """Build the sampling grid for grid_sample under a chosen BEV axis convention.

    The big unknown is how MamBEV's BEV(H=100, W=100) maps to LiDAR (x, y).
    BEVFormer-style heads aren't standardized — different repos pick different
    row/col → x/y mappings, and some flip an axis. Diagnostic 1 tries the
    common candidates and we pick the one with the best warp improvement.

    Conventions probed:
      "cols=x,rows=y"        — what the original analysis used
      "cols=y,rows=x"        — swap (BEVFormer convention if rows index forward)
      "cols=-y,rows=x"       — swap + flip y (some heads use right-handed Z-up)
      "cols=x,rows=-y"       — flip y only

    For each (i_t1, j_t1) output cell of BEV(t+1):
      1. cell center → physical (x_t1, y_t1) in LiDAR_(t+1) under `convention`
      2. Apply T_l_t_from_l_t1 → (x_t, y_t) in LiDAR_t
      3. Map back to normalized (col, row) in BEV(t) under SAME `convention`
    """
    cell = PC_RANGE_LEN / BEV_W   # 1.024 m

    cols = np.arange(BEV_W, dtype=np.float64) + 0.5
    rows = np.arange(BEV_H, dtype=np.float64) + 0.5
    COL, ROW = np.meshgrid(cols, rows, indexing="xy")  # both (H, W)

    # Map (COL, ROW) → physical (x, y) in LiDAR_(t+1) under `convention`
    def grid_to_xy(col, row):
        col_to_phys = PC_RANGE_XY_MIN + col * cell
        row_to_phys = PC_RANGE_XY_MIN + row * cell
        if   convention == "cols=x,rows=y":   return  col_to_phys,  row_to_phys
        elif convention == "cols=y,rows=x":   return  row_to_phys,  col_to_phys
        elif convention == "cols=-y,rows=x":  return  row_to_phys, -col_to_phys + (PC_RANGE_XY_MAX + PC_RANGE_XY_MIN)
        elif convention == "cols=x,rows=-y":  return  col_to_phys, -row_to_phys + (PC_RANGE_XY_MAX + PC_RANGE_XY_MIN)
        raise ValueError(convention)

    Xt1, Yt1 = grid_to_xy(COL, ROW)
    pts_t1 = np.stack([Xt1, Yt1, np.zeros_like(Xt1), np.ones_like(Xt1)], axis=-1)
    pts_t  = pts_t1 @ T_l_t_from_l_t1.T
    Xt, Yt = pts_t[..., 0], pts_t[..., 1]

    # Inverse: given (x_t, y_t) under the same convention, normalized BEV(t) coords
    def xy_to_norm(x, y):
        if   convention == "cols=x,rows=y":
            col = (x - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 - 1.0
            row = (y - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 - 1.0
        elif convention == "cols=y,rows=x":
            col = (y - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 - 1.0
            row = (x - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 - 1.0
        elif convention == "cols=-y,rows=x":
            col = -(y - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 + 1.0
            row =  (x - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 - 1.0
        elif convention == "cols=x,rows=-y":
            col =  (x - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 - 1.0
            row = -(y - PC_RANGE_XY_MIN) / PC_RANGE_LEN * 2.0 + 1.0
        return col, row

    Cn, Rn = xy_to_norm(Xt, Yt)
    grid = np.stack([Cn, Rn], axis=-1).astype(np.float32)  # (H, W, 2)
    return torch.from_numpy(grid).unsqueeze(0).to(device)


# --------------------------------------------------------------------------- #
#  Metrics                                                                    #
# --------------------------------------------------------------------------- #

def per_cell_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Both `(groups, C, H, W)`. Returns `(groups, H, W)`."""
    return torch.sqrt(((a - b) ** 2).sum(dim=1) + 1e-12)


def cosine_per_cell(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    num = (a * b).sum(dim=1)
    da  = torch.sqrt((a * a).sum(dim=1) + eps)
    db  = torch.sqrt((b * b).sum(dim=1) + eps)
    return num / (da * db)


def summarize(name: str, l2: torch.Tensor, cos: torch.Tensor = None):
    mean_l2 = l2.mean().item()
    max_l2  = l2.max().item()
    median_l2 = l2.median().item()
    if cos is not None:
        mean_cos = cos.mean().item()
        frac_pos = (cos > 0.5).float().mean().item()
        print(f"  {name:<14}  L2 mean={mean_l2:.3f}  median={median_l2:.3f}  max={max_l2:.3f}"
              f"   cos mean={mean_cos:.3f}  frac_cos>0.5={frac_pos:.3f}")
    else:
        print(f"  {name:<14}  L2 mean={mean_l2:.3f}  median={median_l2:.3f}  max={max_l2:.3f}")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path,
                        default=Path("/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/bev_pair.pt"))
    args = parser.parse_args()

    print(f"Loading {args.dump} ...")
    d = torch.load(str(args.dump), map_location="cpu", weights_only=False)
    bev_t  = reshape_bev(d["bev_t"].float())        # (groups, C, H, W)
    bev_t1 = reshape_bev(d["bev_t+1"].float())

    print(f"  BEV(t)   : {tuple(bev_t.shape)}  std={bev_t.std().item():.3f}  "
          f"mean={bev_t.mean().item():.3f}")
    print(f"  BEV(t+1) : {tuple(bev_t1.shape)}  std={bev_t1.std().item():.3f}  "
          f"mean={bev_t1.mean().item():.3f}")
    print(f"  scene={d['scene_token'][:8]}, anchors={d['anchor_sample_t'][:8]} → "
          f"{d['anchor_sample_t+1'][:8]}")

    # ---- Warp math ----
    T = compute_warp_T_l_t_from_l_t1(
        eg_t  = d["ego2global_t"],  l2e_t  = d["lidar2ego_t"],
        eg_t1 = d["ego2global_t+1"], l2e_t1 = d["lidar2ego_t+1"],
    )
    # Decompose into translation (m) and yaw (deg) for sanity
    dx, dy, dz = T[:3, 3]
    # Yaw from rotation matrix
    yaw_deg = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    print(f"\nEgo motion t+1 → t (in LiDAR_(t) frame):  "
          f"Δx={dx:+.3f}m  Δy={dy:+.3f}m  Δz={dz:+.3f}m  Δyaw={yaw_deg:+.3f}°")

    # ---- Try all four BEV axis conventions ----
    device = bev_t.device
    zeros  = torch.zeros_like(bev_t1)
    rnd_std = bev_t1.std().item()
    rnd    = torch.randn_like(bev_t1) * rnd_std

    print("\nPer-cell L2 / cosine between baseline and BEV(t+1):")
    print(f"  baseline                       L2_summary                                 cosine_summary")
    summarize("zeros",     per_cell_l2(zeros,  bev_t1), cosine_per_cell(zeros + 1e-9, bev_t1))
    summarize("random",    per_cell_l2(rnd,    bev_t1), cosine_per_cell(rnd,           bev_t1))
    summarize("unwarped",  per_cell_l2(bev_t,  bev_t1), cosine_per_cell(bev_t,         bev_t1))

    warped_results = {}
    mean_l2_unwarp = per_cell_l2(bev_t, bev_t1).mean().item()
    for conv in ("cols=x,rows=y", "cols=y,rows=x",
                 "cols=-y,rows=x", "cols=x,rows=-y"):
        grid = build_grid_sample_grid(T, device, convention=conv)\
                 .expand(bev_t.shape[0], -1, -1, -1)
        bev_t_warped = F.grid_sample(bev_t, grid, mode="bilinear",
                                     padding_mode="zeros", align_corners=False)
        oor = ((grid.abs() > 1.0).any(dim=-1)).float().mean().item()
        l2  = per_cell_l2(bev_t_warped, bev_t1)
        cos = cosine_per_cell(bev_t_warped, bev_t1)
        summarize(f"warped[{conv}]", l2, cos)
        warped_results[conv] = (l2.mean().item(), cos.mean().item(), oor)

    # ---- Compare all conventions to unwarped baseline ----
    print("\nPer-convention warp-vs-unwarped delta (mean L2):")
    best_conv, best_gain = None, -1e9
    for conv, (l2m, cosm, oor) in warped_results.items():
        gain = 100.0 * (mean_l2_unwarp - l2m) / max(mean_l2_unwarp, 1e-9)
        print(f"  {conv:<18}  L2={l2m:.3f}  cos={cosm:.3f}  oor={oor:.3f}  "
              f"gain={gain:+.2f}%")
        if gain > best_gain:
            best_conv, best_gain = conv, gain

    # ---- Detection retention sanity ----
    print("\nTop-5 detection scores per frame (head outputs, no warp / fresh paths):")
    s_t  = d["top20_scores_t"][:5].tolist()
    l_t  = d["top20_labels_t"][:5].tolist()
    s_t1 = d["top20_scores_t+1"][:5].tolist()
    l_t1 = d["top20_labels_t+1"][:5].tolist()
    print(f"  t   : scores={[round(s,3) for s in s_t]}  labels={l_t}")
    print(f"  t+1 : scores={[round(s,3) for s in s_t1]}  labels={l_t1}")

    # ---- Quick interpretation ----
    mean_l2_zeros = per_cell_l2(zeros, bev_t1).mean().item()
    print(f"\nSummary:")
    print(f"  unwarped vs zeros gap : {mean_l2_zeros - mean_l2_unwarp:+.3f}")
    print(f"  best convention       : {best_conv}  ({best_gain:+.2f}% gain vs unwarped)")
    if best_gain > 5.0:
        print("  → ego warp meaningfully closes the BEV gap under this convention.")
        print("    Speculation hypothesis on track; ego-only is the analytical floor.")
    elif best_gain > 0.0:
        print("  → ego warp helps marginally; learned residual needed.")
    else:
        print("  → ego warp does NOT help under ANY axis convention. The BEV is")
        print("    dominated by query-position content, not world-position content.")
        print("    Same wall as Phase 3 / 4 — the encoder cache is what we'd be")
        print("    predicting, and it doesn't have a metric-grid interpretation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
