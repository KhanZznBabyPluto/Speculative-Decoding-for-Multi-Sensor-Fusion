"""Phase 2: per-camera arrival schedules for streaming inference.

StreamPETR runs at 2 Hz nuScenes (one full 6-cam frame every 500 ms). The
*synchronous* assumption is: at query time t·Δt all six cameras have
delivered their image captured at time t. Real systems break this — cams
are independent ROS topics on independent threads with independent
queues, and "all six fresh" is a soft guarantee at best.

This module produces a deterministic schedule:

    arrival[t, c] = the frame index whose CAM_c image is the freshest one
                    available to the system at query time t.

If `arrival[t, c] == t` the camera is fresh; otherwise it is stale by
`t - arrival[t, c]` frames. Generation: per (capture frame f, cam c) we
sample an integer delay d_c(f) ∈ [0, max_delay] uniformly. The image
captured at frame f becomes available at frame f + d_c(f), so at query
time t the freshest cam-c image is

    arrival[t, c] = max { f : f + d_c(f) ≤ t }

Phase 2 uses this to drive a baseline study: how much does the predicted
memory state degrade when the cameras are stale? Phase 3's speculator
will train against the same schedules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NUSC_CAMS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
N_CAMS = len(NUSC_CAMS)


@dataclass
class AsyncSchedule:
    """Per-frame, per-camera arrival index.

    arrival[t, c] ∈ [0, t] is the newest CAM_c frame index available at
    query time t. arrival[t, c] == t means fresh; smaller means stale.
    """

    arrival: np.ndarray         # [T, N_cams] int64
    max_delay: int
    seed: int

    @property
    def T(self) -> int:
        return self.arrival.shape[0]

    @property
    def n_cams(self) -> int:
        return self.arrival.shape[1]

    def staleness(self, t: int) -> np.ndarray:
        """Per-cam delay at frame t, in frame units."""
        return t - self.arrival[t]

    def max_staleness(self, t: int) -> int:
        return int(self.staleness(t).max())

    def is_fresh(self, t: int) -> bool:
        return self.max_staleness(t) == 0

    def last_fully_fresh_frame(self, t: int) -> int:
        """Most recent frame ≤ t at which all cams were available together.

        This is the latest frame the "real" head could have run with
        complete inputs, i.e. our latest dumped-truth state.
        """
        f = t
        while f > 0 and self.arrival[t].min() < f:
            f = int(self.arrival[t].min())
        return f


def make_schedule(T: int, max_delay: int, seed: int,
                  n_cams: int = N_CAMS) -> AsyncSchedule:
    """Construct an AsyncSchedule by sampling per-(capture, cam) delays.

    Per-capture sampling (not per-query) keeps the arrival sequence
    monotone-non-decreasing per camera, which matches reality: once
    cam-c's frame f arrives, it stays available; later queries either
    see frame f or something newer.
    """
    rng = np.random.default_rng(seed)
    delays = rng.integers(0, max_delay + 1, size=(T, n_cams))  # [T, N]
    avail_time = np.arange(T)[:, None] + delays                # [T, N]

    arrival = np.zeros((T, n_cams), dtype=np.int64)
    for t in range(T):
        # For each cam c: max f in [0..t] s.t. avail_time[f, c] ≤ t.
        valid = avail_time[: t + 1] <= t                       # [t+1, N]
        f_idx = np.arange(t + 1)[:, None]                      # [t+1, 1]
        masked = np.where(valid, f_idx, -1)
        arrival[t] = masked.max(axis=0)
        # Frame 0 must always have *something* — treat negative as 0 (i.e.
        # use the capture even though "officially" it hasn't arrived yet).
        # In practice this only kicks in if delays[0] > 0; we accept the
        # tiny self-cheat at the boundary.
        arrival[t] = np.maximum(arrival[t], 0)
    return AsyncSchedule(arrival=arrival, max_delay=max_delay, seed=seed)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test():
    # max_delay=0 → fully synchronous: arrival == arange(T) per cam.
    s0 = make_schedule(T=10, max_delay=0, seed=0)
    assert (s0.arrival == np.arange(10)[:, None]).all()
    assert s0.is_fresh(5)
    assert s0.max_staleness(5) == 0
    print("[OK] max_delay=0 ⇒ synchronous schedule")

    # max_delay≥1: arrival is monotone-non-decreasing per cam, ≤ t everywhere.
    s = make_schedule(T=50, max_delay=2, seed=42)
    for c in range(s.n_cams):
        col = s.arrival[:, c]
        assert (np.diff(col) >= 0).all(), f"cam {c}: arrival decreased"
        assert (col <= np.arange(50)).all(), f"cam {c}: arrival exceeded t"
    print("[OK] arrival is monotone-non-decreasing and bounded by t")

    # Staleness distribution: expected mean ≈ max_delay / 2 in steady state.
    stalls = np.stack([s.staleness(t) for t in range(20, 50)])   # [30, N]
    mean_stale = stalls.mean()
    assert 0.5 < mean_stale < 1.5, f"steady-state mean staleness {mean_stale}"
    print(f"[OK] steady-state mean staleness ≈ {mean_stale:.2f} (≈1.0 expected)")

    # Determinism: same seed ⇒ same schedule.
    s_again = make_schedule(T=50, max_delay=2, seed=42)
    assert (s.arrival == s_again.arrival).all()
    print("[OK] schedule generation is deterministic w.r.t. seed")

    print("All async_schedule self-tests pass.")


if __name__ == "__main__":
    _self_test()
