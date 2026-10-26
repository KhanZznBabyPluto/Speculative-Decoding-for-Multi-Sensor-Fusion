"""Phase 1.2: verify our `latent_ops.warp_memory_*` matches StreamPETR's
native warp semantics on a real dumped scene.

Three checks:

  (A) Round-trip identity. For each frame, warp memory from global → ego_t
      → global. Result must equal the original to floating-point noise.

  (B) Cross-frame consistency. For a memory token that exists at both
      frame t and frame t+1 (i.e. one of the prepended-once-and-still-here
      tokens), its reference point in GLOBAL frame should be the same
      across the two snapshots up to ego-frame->global rounding.
      (Reference points are stored in global frame between frames, so
      they shouldn't drift just from ego motion.)

  (C) Native-warp match. Take state at frame t (global), warp it into the
      ego frame of frame t+1 using our helper. Verify the result is in a
      reasonable BEV range (norm ≤ a few hundred meters), which is the
      sanity test that ego_pose_inv was applied correctly.

Run:
    python stages/04_streampetr_latent/verify_warp.py \
        stages/04_streampetr_latent/dumps/scene_0.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from latent_ops import MemoryState, warp_memory_to_ego, warp_memory_to_global


def _state_from_frame(frame: dict, frame_label: str = "global") -> MemoryState:
    """Build a B=1 MemoryState from a dump-frame dict."""
    return MemoryState(
        embedding=frame["memory_embedding"].unsqueeze(0),
        reference_point=frame["memory_reference_point"].unsqueeze(0),
        timestamp=frame["memory_timestamp"].unsqueeze(0),
        egopose=frame["memory_egopose"].unsqueeze(0),
        velo=frame["memory_velo"].unsqueeze(0),
        frame=frame_label,
    )


def _ego_inv(ego_pose: np.ndarray) -> torch.Tensor:
    inv = np.linalg.inv(ego_pose)
    return torch.from_numpy(inv).float().unsqueeze(0)  # [1, 4, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help=".pt produced by dump_latents.py")
    args = ap.parse_args()

    dump = torch.load(args.dump, map_location="cpu")
    scene_tok = list(dump.keys())[0]
    frames = dump[scene_tok]["frames"]
    print(f"Loaded {len(frames)} frames from scene {scene_tok[:8]}")

    # (A) Round-trip on every frame -------------------------------------
    print("\n[A] Round-trip identity (global → ego_t → global) per frame:")
    max_err = 0.0
    for i, f in enumerate(frames):
        ego_pose = torch.from_numpy(f["ego_pose"]).float().unsqueeze(0)
        ego_pose_inv_t = ego_pose.inverse()
        s_glob = _state_from_frame(f, frame_label="global")
        s_ego = warp_memory_to_ego(s_glob, ego_pose_inv_t)
        s_back = warp_memory_to_global(s_ego, ego_pose)
        err = (s_back.reference_point - s_glob.reference_point).abs().max().item()
        max_err = max(max_err, err)
        if i in (0, 5, 10, 20, 39):
            print(f"  frame {i:2d}: max |ref - round-trip(ref)| = {err:.2e}")
    print(f"  → worst frame error: {max_err:.2e}  "
          + ("OK" if max_err < 1e-2 else "FAIL"))

    # (B) Cross-frame ref consistency -----------------------------------
    # Tokens stored in global frame should not drift between consecutive
    # frames just because ego moved. We check the OLDEST tokens (the ones
    # that survived from frame t-1 into frame t at slots [num_propagated:]).
    print("\n[B] Cross-frame consistency of GLOBAL-frame ref points (frames 5↔6):")
    f5, f6 = frames[5], frames[6]
    ref5 = f5["memory_reference_point"]      # [1280, 3]
    ref6 = f6["memory_reference_point"]      # [1280, 3]
    # In the new frame's memory, the first num_propagated=256 entries are
    # the NEW tokens prepended by post_update. The next memory_len=1024
    # are the carried-over older tokens. The carried-over tokens at slot
    # [256+k] in frame 6 correspond to tokens at slot [k] in frame 5
    # (after memory_refresh trimming).
    num_propagated = 256
    older_in_6 = ref6[num_propagated:num_propagated + 1024]
    same_in_5 = ref5[: 1024]
    delta = (older_in_6 - same_in_5).abs().max().item()
    print(f"  max |ref_5[0:1024] - ref_6[256:1280]| = {delta:.4f} m  "
          + ("OK" if delta < 1.0 else "FAIL"))

    # (C) Native-warp sanity: ref points after warp into ego_{t+1} are
    # within the BEV range (roughly ±100 m).
    print("\n[C] Warp from frame 5 (global) into ego frame of frame 6:")
    ego_pose_inv_6 = _ego_inv(f6["ego_pose"])
    s5 = _state_from_frame(f5, frame_label="global")
    s5_in_ego6 = warp_memory_to_ego(s5, ego_pose_inv_6)
    real_mask = s5_in_ego6.reference_point.norm(dim=-1) > 1e-6
    ref_in_ego = s5_in_ego6.reference_point[real_mask]
    max_xy = ref_in_ego[:, :2].abs().max().item()
    print(f"  max |xy| of warped real tokens: {max_xy:.1f} m  "
          + ("OK" if max_xy < 500.0 else "FAIL"))
    print(f"  warped state.frame = {s5_in_ego6.frame!r}  "
          + ("OK" if s5_in_ego6.frame == 'ego' else 'FAIL'))


if __name__ == "__main__":
    main()
