"""Ego-motion BEV warp utility.

Phase A of stage 06 needs to warp a dense BEV tensor [B, C, H, W] from
ego-frame `t-1` into ego-frame `t`, given the global ego poses at both
times. No learnable params.

The math (BEVFormer's `obtain_history_bev` does the same thing; we
reimplement it here because we want it standalone, without going through
the head's API):

  1. Each BEV cell (i, j) maps to a 2D point in the *current* ego frame,
     centered on the ego origin, with x = forward, y = left.
  2. Compose the inter-frame ego SE(3) (the 2D part) to find where that
     same world point sat in the *previous* ego frame.
  3. Convert back to (u, v) normalized coords for grid_sample.

Conventions:
  - BEV grid is centered on the ego (the ego origin sits at the *center*
    of the (H, W) tensor, not at a corner).
  - real_h, real_w are the metric BEV extents (e.g., 102.4 m × 102.4 m
    for many BEVFormer configs, 100 m × 100 m for some). Read these from
    the MamBEV config and pass them in — DO NOT hard-code.
  - Ego pose convention: ego2global is a 4x4 homogeneous matrix in
    `[forward, left, up]` ego coords. This matches nuScenes-devkit when
    the rotation is built from the LIDAR_TOP `ego_pose` quaternion +
    translation (the same construction used in stage 04
    dump_latents.py:57-69, after the pyquaternion fix).

Sanity-tested below in `_self_test()` with CPU-only synthetic poses.
Run as `python bev_warp.py` to execute the self-test.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BEVGrid:
    """Metadata describing one BEV tensor's spatial layout."""
    h: int            # number of BEV rows (e.g., 200 or 180)
    w: int            # number of BEV cols
    real_h: float     # metric extent in y (left) direction, metres
    real_w: float     # metric extent in x (forward) direction, metres


def _make_cell_centers(grid: BEVGrid, device, dtype) -> torch.Tensor:
    """Return [H, W, 2] tensor of (x_forward, y_left) metric coords for
    each BEV cell, centered on ego origin (so cell (H/2, W/2) is at 0,0)."""
    # cell (row=i, col=j); row index increases "forward" (x), col index
    # increases "left" (y). This convention matches BEVFormer's
    # spatial_cross_attention layout.
    dx = grid.real_w / grid.w
    dy = grid.real_h / grid.h
    # +0.5 so coords are cell-centers, not cell-corners.
    xs = (torch.arange(grid.w, device=device, dtype=dtype) + 0.5) * dx \
        - grid.real_w / 2.0
    ys = (torch.arange(grid.h, device=device, dtype=dtype) + 0.5) * dy \
        - grid.real_h / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")     # [H, W]
    return torch.stack([xx, yy], dim=-1)               # [H, W, 2]


def warp_bev(
    bev_prev: torch.Tensor,           # [B, C, H, W]
    ego2global_prev: torch.Tensor,    # [B, 4, 4]
    ego2global_curr: torch.Tensor,    # [B, 4, 4]
    grid: BEVGrid,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp BEV from ego-frame (prev) into ego-frame (curr) via ego-motion.

    Returns a tensor of the same shape as `bev_prev`. Cells whose
    pre-image falls outside the previous BEV extent are zeroed (with
    padding_mode="zeros"). Use padding_mode="border" if you'd rather
    extrapolate.

    NOTE: this is currently a *2D* warp (we drop z entirely). For BEV
    that is the right thing — height is collapsed before this function
    is called. If you ever need 3D, this is the wrong utility.
    """
    B, C, H, W = bev_prev.shape
    assert (H, W) == (grid.h, grid.w), \
        f"bev shape {(H, W)} doesn't match BEVGrid {(grid.h, grid.w)}"
    device, dtype = bev_prev.device, bev_prev.dtype

    # 1. Cell centers in *current* ego frame (metric, ego-origin centered).
    pts_curr_ego = _make_cell_centers(grid, device, dtype)         # [H, W, 2]

    # 2. Lift to homogeneous 4D (z=0, w=1) and transform:
    #    pt_world = ego2global_curr @ pt_curr_ego
    #    pt_prev_ego = inv(ego2global_prev) @ pt_world
    pts4 = torch.cat([
        pts_curr_ego,
        torch.zeros_like(pts_curr_ego[..., :1]),
        torch.ones_like(pts_curr_ego[..., :1]),
    ], dim=-1)                                                     # [H, W, 4]
    pts4 = pts4.view(1, H * W, 4).expand(B, -1, -1)                # [B, H*W, 4]

    pts_world = torch.bmm(pts4, ego2global_curr.transpose(1, 2))   # [B, H*W, 4]
    inv_prev = torch.linalg.inv(ego2global_prev.to(torch.float64)).to(dtype)
    pts_prev_ego = torch.bmm(pts_world, inv_prev.transpose(1, 2))  # [B, H*W, 4]
    xs_prev = pts_prev_ego[..., 0]                                  # [B, H*W]
    ys_prev = pts_prev_ego[..., 1]

    # 3. Convert metric (x_forward, y_left) back to grid_sample uv in [-1, 1].
    #    In our convention x↔col, y↔row, with origin at tensor center.
    u = (xs_prev / (grid.real_w / 2.0))
    v = (ys_prev / (grid.real_h / 2.0))
    sample_grid = torch.stack([u, v], dim=-1).view(B, H, W, 2)

    return F.grid_sample(
        bev_prev, sample_grid,
        mode=mode, padding_mode=padding_mode, align_corners=False,
    )


# ---------------------------------------------------------------------------
# CPU self-tests
# ---------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    grid = BEVGrid(h=200, w=200, real_h=102.4, real_w=102.4)
    B, C = 1, 4
    bev = torch.randn(B, C, grid.h, grid.w)

    # (1) Identity ego pose → identity warp → bit-exact recovery.
    I = torch.eye(4).unsqueeze(0)
    out = warp_bev(bev, I, I, grid, padding_mode="border")
    err = (out - bev).abs().max().item()
    print(f"[bev_warp self-test] identity ego pose: max |err| = {err:.6f}")
    assert err < 1e-5, "Identity warp should be exact."

    # (2) Pure translation (1 m forward = +x). Warp should shift content
    #     "backward" in row direction by 1m / (real_w / w) ≈ 1.95 cells.
    T = torch.eye(4).unsqueeze(0)
    T[0, 0, 3] = 1.0   # translate prev->global by +1m forward
    out = warp_bev(bev, ego2global_prev=T, ego2global_curr=I, grid=grid)
    # Just sanity-check that the warp moved *something* and didn't blow up.
    diff = (out - bev).abs().mean().item()
    print(f"[bev_warp self-test] +1 m translation: mean |out-bev| = {diff:.4f}")
    assert diff > 0.0, "Translation should change the content."

    # (3) Round trip: warp prev→curr then curr→prev with same poses returns
    #     close-to-original (modulo grid_sample interpolation losses).
    out_fwd = warp_bev(bev, T, I, grid, padding_mode="border")
    out_back = warp_bev(out_fwd, I, T, grid, padding_mode="border")
    rt_err = (out_back - bev).abs().mean().item()
    print(f"[bev_warp self-test] round-trip mean |err| = {rt_err:.4f}")
    # Interpolation will smear; tolerance is loose.
    assert rt_err < 0.5, "Round-trip blew up — coordinate convention bug?"

    print("[bev_warp self-test] OK")


if __name__ == "__main__":
    _self_test()
