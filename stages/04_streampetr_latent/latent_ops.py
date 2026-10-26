"""Sparse-memory operations for StreamPETR latent state.

StreamPETR's persistent scene state is NOT a dense BEV grid. It lives on
`pts_bbox_head` as five tensors per batch element:

    memory_embedding        [B, M, D]      query tokens
    memory_reference_point  [B, M, 3]      3D position per token (global frame
                                            between calls; ego frame inside
                                            the head's forward)
    memory_timestamp        [B, M, 1]
    memory_egopose          [B, M, 4, 4]
    memory_velo             [B, M, 2]

Between consecutive frames, the head stores the reference points in the
GLOBAL frame so they survive ego motion. Inside a forward pass the head:

    pre_update:   memory_reference_point = ego_pose_inv @ memory_reference_point
                  memory_egopose         = ego_pose_inv @ memory_egopose
                  memory_timestamp      += data['timestamp']

    post_update:  memory_reference_point = ego_pose @ memory_reference_point
                  memory_egopose         = ego_pose @ memory_egopose
                  memory_timestamp      -= data['timestamp']

`ego_pose` here is the ego→global transform at the current frame;
`ego_pose_inv` is its inverse (global→ego at the current frame).

This module provides the same primitives independent of the head, so:
  * we can warp dumped memory state to any frame for the speculator
  * we can unit-test warp correctness without GPU / mmcv
  * Phase 3's speculator (z_{t+1} = z_t + f(...)) can operate cleanly
    on a well-typed state object
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass
class MemoryState:
    """A snapshot of StreamPETR's persistent memory.

    All tensors are batched with a leading B dim (typically B=1 at test).
    `frame`:
        - "global"  : `reference_point` and `egopose` are in global coords.
                     This is how StreamPETR stores memory BETWEEN forward calls.
        - "ego_t"   : `reference_point` and `egopose` are in the ego frame at
                     timestep t. This is what the head sees DURING forward.
    """

    embedding: torch.Tensor          # [B, M, D]
    reference_point: torch.Tensor    # [B, M, 3]
    timestamp: torch.Tensor          # [B, M, 1]
    egopose: torch.Tensor            # [B, M, 4, 4]
    velo: torch.Tensor               # [B, M, 2]
    frame: str = "global"
    sample_token: Optional[str] = None

    def to(self, device) -> "MemoryState":
        return replace(
            self,
            embedding=self.embedding.to(device),
            reference_point=self.reference_point.to(device),
            timestamp=self.timestamp.to(device),
            egopose=self.egopose.to(device),
            velo=self.velo.to(device),
        )

    def clone(self) -> "MemoryState":
        return replace(
            self,
            embedding=self.embedding.clone(),
            reference_point=self.reference_point.clone(),
            timestamp=self.timestamp.clone(),
            egopose=self.egopose.clone(),
            velo=self.velo.clone(),
        )


# ---------------------------------------------------------------------------
# Point + memory warps (mirror StreamPETR's misc.transform_reference_points)
# ---------------------------------------------------------------------------
def transform_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 rigid transform to a batch of 3D points.

    points: [..., 3]
    matrix: [..., 4, 4]  (broadcastable to points' batch dims)
    returns: [..., 3]

    Matches StreamPETR's misc.transform_reference_points(reverse=False,
    translation=True) bit-for-bit.
    """
    ones = torch.ones_like(points[..., :1])
    homo = torch.cat([points, ones], dim=-1)                # [..., 4]
    # Broadcast matrix to points' shape: matrix has one fewer trailing dim.
    out = (matrix @ homo.unsqueeze(-1)).squeeze(-1)         # [..., 4]
    return out[..., :3]


def warp_memory_to_ego(state: MemoryState, ego_pose_inv: torch.Tensor) -> MemoryState:
    """Bring memory (currently in global frame) into the current ego frame.

    ego_pose_inv: [B, 4, 4] = global→ego at the current frame.
    Mirrors StreamPETR's pre_update_memory (lines 331-332).
    """
    assert state.frame == "global", \
        f"warp_memory_to_ego expects state.frame=='global', got {state.frame!r}"
    epi = ego_pose_inv.unsqueeze(1)  # [B, 1, 4, 4] — broadcast over M tokens
    new_ref = transform_points(state.reference_point, epi)
    new_egopose = epi @ state.egopose
    return replace(state, reference_point=new_ref, egopose=new_egopose, frame="ego")


def warp_memory_to_global(state: MemoryState, ego_pose: torch.Tensor) -> MemoryState:
    """Send memory (currently in ego_t frame) back into global frame.

    ego_pose: [B, 4, 4] = ego→global at the current frame.
    Mirrors StreamPETR's post_update_memory (lines 372-374).
    """
    assert state.frame == "ego", \
        f"warp_memory_to_global expects state.frame=='ego', got {state.frame!r}"
    ep = ego_pose.unsqueeze(1)
    new_ref = transform_points(state.reference_point, ep)
    new_egopose = ep @ state.egopose
    return replace(state, reference_point=new_ref, egopose=new_egopose, frame="global")


def step_memory(
    state_at_t: MemoryState,
    ego_pose_t: torch.Tensor,
    ego_pose_inv_tp1: torch.Tensor,
    dt: torch.Tensor,
) -> MemoryState:
    """Single-step warp: ego frame at t → ego frame at t+1, with timestamp update.

    Convenience wrapper: equivalent to (to_global at t → to_ego at t+1).
    Useful for the Phase 3 speculator which operates frame-to-frame.

    ego_pose_t:        [B, 4, 4]  ego→global at time t
    ego_pose_inv_tp1:  [B, 4, 4]  global→ego at time t+1
    dt:                [B, 1, 1]  positive number of seconds elapsed
    """
    s_global = warp_memory_to_global(state_at_t, ego_pose_t)
    s_global = replace(s_global, timestamp=s_global.timestamp + dt)
    return warp_memory_to_ego(s_global, ego_pose_inv_tp1)


# ---------------------------------------------------------------------------
# Self-tests (CPU, no mmcv) — run as `python latent_ops.py`
# ---------------------------------------------------------------------------
def _se3_rotz(theta: float) -> torch.Tensor:
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    return torch.tensor([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _se3_trans(tx: float, ty: float) -> torch.Tensor:
    T = torch.eye(4)
    T[0, 3] = tx
    T[1, 3] = ty
    return T


def _self_test():
    torch.manual_seed(0)
    B, M, D = 1, 5, 32

    # Random memory in global frame
    s = MemoryState(
        embedding=torch.randn(B, M, D),
        reference_point=torch.randn(B, M, 3) * 10.0,
        timestamp=torch.zeros(B, M, 1),
        egopose=torch.eye(4).expand(B, M, 4, 4).clone(),
        velo=torch.zeros(B, M, 2),
        frame="global",
    )

    # Pretend the ego moved +5 m in x and rotated 0.1 rad since some reference.
    ego_pose = (_se3_trans(5.0, 0.0) @ _se3_rotz(0.1)).unsqueeze(0)  # [1, 4, 4]
    ego_pose_inv = ego_pose.inverse()

    # Round-trip: global → ego → global should be identity.
    s_ego = warp_memory_to_ego(s, ego_pose_inv)
    s_back = warp_memory_to_global(s_ego, ego_pose)
    assert torch.allclose(s_back.reference_point, s.reference_point, atol=1e-5), \
        f"round-trip ref_point diff = {(s_back.reference_point - s.reference_point).abs().max()}"
    assert torch.allclose(s_back.egopose, s.egopose, atol=1e-5)
    assert s_back.frame == "global" and s_ego.frame == "ego"
    print("[OK] global → ego → global round-trip preserves state")

    # step_memory: at t=0 (frame ego_0), step to t=1 (frame ego_1).
    # If ego doesn't move, the memory should be unchanged (modulo timestamp).
    s_at_t = warp_memory_to_ego(s, ego_pose_inv)            # in ego_0
    identity = torch.eye(4).unsqueeze(0)
    dt = torch.tensor([[[0.5]]])
    s_at_tp1 = step_memory(s_at_t, ego_pose, ego_pose.inverse(), dt)
    # Same ego_pose used for both => no relative motion => ref_point unchanged.
    assert torch.allclose(s_at_tp1.reference_point, s_at_t.reference_point, atol=1e-5)
    assert torch.allclose(s_at_tp1.timestamp, s_at_t.timestamp + 0.5)
    print("[OK] step_memory with zero ego motion preserves positions, advances time")

    # Sanity: a pure translation of ego by +5 m in x should shift memory ref
    # points by -5 m in the ego frame (the world appears to move backward
    # relative to ego).
    s_at_0 = MemoryState(
        embedding=torch.randn(B, M, D),
        reference_point=torch.tensor([[[0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [10.0, 0.0, 0.0],
                                        [0.0, 10.0, 0.0]]]),
        timestamp=torch.zeros(B, M, 1),
        egopose=torch.eye(4).expand(B, M, 4, 4).clone(),
        velo=torch.zeros(B, M, 2),
        frame="ego",
    )
    ego_t   = torch.eye(4).unsqueeze(0)
    ego_tp1 = _se3_trans(5.0, 0.0).unsqueeze(0)
    s_at_1 = step_memory(s_at_0, ego_t, ego_tp1.inverse(), dt)
    # The reference at (10, 0, 0) in ego_0 was global (10, 0, 0); ego moved to
    # (5, 0, 0); in ego_1 the point is at (10-5, 0, 0) = (5, 0, 0).
    expected = torch.tensor([[[-5.0, 0.0, 0.0],
                               [-4.0, 0.0, 0.0],
                               [-5.0, 1.0, 0.0],
                               [ 5.0, 0.0, 0.0],
                               [-5.0, 10.0, 0.0]]])
    assert torch.allclose(s_at_1.reference_point, expected, atol=1e-5), \
        f"got {s_at_1.reference_point}, expected {expected}"
    print("[OK] ego translation of +5 m shifts memory ref by -5 m in new ego")

    print("All latent_ops self-tests pass.")


if __name__ == "__main__":
    _self_test()
