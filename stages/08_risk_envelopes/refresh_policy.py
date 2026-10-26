"""When must the perception stack actually recompute? (zero GPU)

Stage 08 so far bounds where an object is, given that the frame is k steps old.
The systems question is the other direction: k is not fate, it is a CHOICE. A
stack can spend a full forward pass and reset k to 0, or coast and let k grow.
This script prices that choice.

WHAT MAKES THIS MEASURABLE WITHOUT A GPU. Detections already exist on disk for
k = 0, 1, 2, 3 (fresh plus three honest-key delay runs). A refresh policy is
then just an assignment of a staleness k to every frame, and the emitted
envelope set for that frame is read off the matching config. No inference.

THE COST AXIS. Conformal calibration pins coverage at 1 - alpha for every group,
so a policy cannot buy coverage by refreshing more — it buys SPACE. The
deliverable is therefore claimed area against refresh rate, and the question is
which policy holds the tightest bound per forward pass spent.

WHY THE RADIUS IS CALIBRATED PER (k, EGO DISPLACEMENT). The 2026-07-17 session
established that staleness damage scales with how far the ego moved, not with
elapsed time as such: retention was 85.8 % below 0.5 m of ego motion and 11.5 %
above 4 m. Calibrating on k alone would give every k=2 frame the same radius and
leave an adaptive policy nothing to exploit. Conditioning on the ego
displacement the policy can actually observe (odometry is available at 100-200
Hz, long before the next camera frame) is what lets "coast while stopped, refresh
while moving" show up as a smaller bound rather than as nothing.

THE POLICIES, all zero-parameter and all computable at run time:

    periodic(N)   refresh every N-th frame. The baseline any scheduler beats or
                  does not.
    ego(d)        refresh once the ego has travelled d metres since the last
                  refresh. Uses odometry only.
    obj(d)        refresh once the FASTEST object seen at the last refresh could
                  have moved d metres. Uses only the predicted velocities of the
                  detections already emitted, so it is available at run time.
    random(p)     refresh with probability p. The control that says whether a
                  policy is doing better than spending the same budget blindly.

WHY `obj` EXISTS, and it is a correction to this script's own first design.
`ego` was the mechanism-driven candidate, because the 2026-07-17 session showed
damage scaling with ego displacement. Measured here, it LOSES to periodic, and
the calibration table says why: at k=1 the radius is 3.48 m below 0.5 m of ego
motion but 2.51 m between 0.5 and 2 m — non-monotone. The honest key-side lift
has already removed the ego term (it was measured to be "inert at zero ego
motion, helping in proportion to displacement"), so on the corrected system the
residual error is OBJECT motion, which odometry cannot predict. Ego displacement
stratified the error of the naive harness, not of this one. `obj` is the trigger
that hypothesis actually implies.

THE CAVEAT THAT BOUNDS ALL OF THIS, stated up front. The k=2 detections come
from a run where EVERY frame used 2-frame-old features, not from a run that
refreshed and then coasted for two frames. The photon age is right; the memory
history is not — StreamPETR carries `memory_embedding` forward, so a real
policy's state at a coasting frame differs from ours. This is the same
trajectory-mismatch that produced the Phase 4 chimera, and it means these
numbers are a first-order estimate of a policy's cost, not a simulation of one.
Closing it needs a run with the policy in the loop (GPU), which is the natural
follow-up.

Usage:
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/08_risk_envelopes/refresh_policy.py
"""

import argparse
import json
import os
import time

import numpy as np

from conformal_envelopes import (CLASS_NAMES, conformal_quantile, make_splits)
from envelope_coverage import NUSC_DT
from envelope_precision import BETA, RHO, score_pass, space_metrics
from envelope_coverage import (CONFIGS, build_gt_table, load_ground_truth,
                               load_nusc, load_predictions, OUT_DIR)
from eval_subset import frame_index_in_scene, load_infos, token_to_scene

# ── Config ────────────────────────────────────────────────────────────────

KS = (0, 1, 2, 3)
CONFIG_FOR_K = {0: "fresh", 1: "d1_key", 2: "d2_key", 3: "d3_key"}
MAX_K = 3                       # no detections exist beyond this; policies clamp

CLASSES = ("car",)
BUDGET = 50                     # the operating point from the precision sweep
ALPHA = 0.1
N_SPLITS = 20
N_AREA_SPLITS = 4                 # was 2; per-split areas are kept for paired tests

# Ego displacement (metres, since the frame the photons came from) defining the
# calibration groups. The first bucket is the near-static regime where staleness
# was measured to be nearly harmless.
EGO_BINS = (0.0, 0.5, 2.0, 4.0, np.inf)

PERIODS = (1, 2, 3, 4)
EGO_THRESHOLDS = (0.5, 1.5, 3.0, 6.0, 12.0)
OBJ_THRESHOLDS = (1.0, 2.5, 5.0, 10.0, 20.0)
RANDOM_RATES = (1.0, 0.5, 0.34, 0.25)
RNG_SEED = 0
N_RANDOM_SEEDS = 3                # random policy averaged over this many draws


# ── Ego motion bookkeeping ────────────────────────────────────────────────

def ego_track(by_scene):
    """Per-scene ordered token lists plus cumulative ego path length.

    Returns:
        dict scene_token -> (tokens, xy) where xy is (T, 2) ego position in the
        global frame, in temporal order.
    """
    out = {}
    for st, frames in by_scene.items():
        toks = [f["token"] for f in frames]
        xy = np.array([f["ego2global_translation"][:2] for f in frames],
                      dtype=np.float64)
        out[st] = (toks, xy)
    return out


def ego_gap(xy, i, j):
    """Straight-line ego displacement between frame indices j and i."""
    return float(np.linalg.norm(xy[i] - xy[j]))


def bucket_of(disp):
    """Index of the EGO_BINS bucket containing `disp`."""
    return int(np.searchsorted(np.asarray(EGO_BINS[1:]), disp, side="right"))


# ── Policies ──────────────────────────────────────────────────────────────

def max_object_speed(pred, budget):
    """Fastest predicted object speed per frame, from the emitted detections.

    This is what a running system already has in hand — the velocities it just
    published — so a trigger built on it needs no extra sensing.

    Returns:
        dict token -> float m/s (0.0 where the frame has no detection).
    """
    out = {}
    for tok, p in pred.items():
        v = p["velo"][:budget]
        out[tok] = float(np.linalg.norm(v, axis=1).max()) if len(v) else 0.0
    return out


def run_policy(kind, param, toks, xy, rng, obj_speed=None):
    """Assign a staleness k to every frame of one scene.

    Frame 0 is always a refresh (nothing to coast from), and k is clamped at
    MAX_K because no deeper detections exist — a policy that would have coasted
    longer is forced to refresh, which makes its measured refresh rate an
    UPPER bound on what it would actually spend.

    Returns:
        (k_per_frame, refreshed) both length T; refreshed is a bool array.
    """
    T = len(toks)
    k = np.zeros(T, dtype=np.int32)
    refreshed = np.zeros(T, dtype=bool)
    last = 0
    refreshed[0] = True
    for i in range(1, T):
        age = i - last
        if kind == "periodic":
            do = (age >= param)
        elif kind == "ego":
            do = (ego_gap(xy, i, last) >= param)
        elif kind == "obj":
            # How far the fastest object seen at the last refresh could have
            # travelled since; all of it known at the last refresh.
            v = obj_speed.get(toks[last], 0.0) if obj_speed else 0.0
            do = (v * age * NUSC_DT >= param)
        elif kind == "random":
            do = (rng.random() < param)
        else:
            raise ValueError(kind)
        if age >= MAX_K:          # forced: no data beyond MAX_K
            do = True
        if do:
            last = i
            refreshed[i] = True
            k[i] = 0
        else:
            k[i] = i - last
    return k, refreshed


# ── Main ──────────────────────────────────────────────────────────────────

def run(out_json, do_area):
    """Calibrate per (k, ego bucket), then price each policy by claimed area."""
    nusc, cfg = load_nusc()
    by_scene = load_infos()
    tok2scene = token_to_scene(by_scene)
    fidx = frame_index_in_scene(by_scene)
    ego_xy = {i["token"]: np.asarray(i["ego2global_translation"][:2], np.float64)
              for fs in by_scene.values() for i in fs}
    tracks = ego_track(by_scene)

    ref = load_predictions(nusc, cfg, "fresh")
    tokens = sorted(ref.keys())
    gt_all = load_ground_truth(nusc, cfg, tokens)

    ids = [CLASS_NAMES.index(c) for c in CLASSES]

    def keep_cls(boxes):
        """Restrict every per-token array dict to the classes under study."""
        out = {}
        for tok, b in boxes.items():
            sel = np.isin(b["cls"], ids)
            out[tok] = {kk: vv[sel] for kk, vv in b.items()
                        if isinstance(vv, np.ndarray)}
        return out

    gt = keep_cls(gt_all)
    table, spans, scenes = build_gt_table(gt, tok2scene, fidx)
    n_rows = len(table["cls"])
    scene_list = sorted(set(tok2scene.values()))
    for tok in gt:
        gt[tok]["scene_idx"] = scene_list.index(tok2scene[tok])
    print(f"[setup] {n_rows} car GT objects, {len(scene_list)} scenes, "
          f"budget={BUDGET}, alpha={ALPHA}", flush=True)

    # ── per-k detections and scores ──
    preds, gt_scores = {}, {}
    for k in KS:
        p = load_predictions(nusc, cfg, CONFIG_FOR_K[k], tokens)
        preds[k] = keep_cls(p)
        gt_scores[k], *_ = score_pass(gt, preds[k], spans, n_rows, k, BUDGET)
        print(f"  k={k} ({CONFIG_FOR_K[k]}): scored", flush=True)

    # Ego displacement over the staleness gap, per token per k, and the bucket.
    tok_bucket = {}
    for k in KS:
        for st, (toks, xy) in tracks.items():
            for i, t in enumerate(toks):
                j = max(i - k, 0)
                tok_bucket[(k, t)] = bucket_of(ego_gap(xy, i, j))

    # GT rows carry the bucket of their own frame, per k.
    gt_tok = np.empty(n_rows, dtype=object)
    for tok, (lo, hi) in spans.items():
        gt_tok[lo:hi] = tok
    gt_bucket = {k: np.array([tok_bucket[(k, t)] for t in gt_tok], dtype=np.int32)
                 for k in KS}

    splits = make_splits(len(scene_list), n_splits=N_SPLITS)
    n_buckets = len(EGO_BINS) - 1

    # ── Mondrian calibration: one radius per (k, ego bucket) ──
    qs = []
    for cal, _ in splits:
        q = np.full((len(KS), n_buckets), np.inf)
        cal_m = np.isin(table["scene_idx"], cal)
        for k in KS:
            for b in range(n_buckets):
                # Frames with index < k fell back to fresh features in the
                # k-config dump; they are not k-stale and must not calibrate it.
                sel = cal_m & (gt_bucket[k] == b) & (table["frame_idx"] >= k)
                if sel.sum() >= 20:        # too few to certify anything
                    q[k, b] = conformal_quantile(gt_scores[k][sel], ALPHA)
        qs.append(q)
    q_med = np.nanmedian(np.where(np.isinf(qs), np.nan, qs), axis=0)
    print("\nCalibrated radius (m) per (k, ego displacement since capture):")
    print(f"{'k':>3} " + "".join(f"{f'[{EGO_BINS[b]:.1f},{EGO_BINS[b+1]:.1f})':>14}"
                                 for b in range(n_buckets)))
    for k in KS:
        print(f"{k:>3} " + "".join(f"{q_med[k, b]:>14.2f}"
                                   for b in range(n_buckets)))

    # ── policies ──
    # The trigger reads the fresh detections, which is what a system that just
    # refreshed would be holding.
    obj_speed = max_object_speed(preds[0], BUDGET)

    policies = ([("periodic", n) for n in PERIODS]
                + [("ego", d) for d in EGO_THRESHOLDS]
                + [("obj", d) for d in OBJ_THRESHOLDS]
                + [("random", p) for p in RANDOM_RATES])

    rows = []
    for kind, param in policies:
        t0 = time.time()
        # The random control is one realisation per seed; average several so
        # its ranking against the deterministic policies is not a single draw.
        seeds = range(RNG_SEED, RNG_SEED + (N_RANDOM_SEEDS if kind == "random" else 1))
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            k_of_tok, refresh_flags = {}, []
            for st, (toks, xy) in tracks.items():
                kk, rf = run_policy(kind, param, toks, xy, rng, obj_speed)
                for t, a, b in zip(toks, kk, rf):
                    k_of_tok[t] = int(a)
                    refresh_flags.append(bool(b))
            rate = float(np.mean(refresh_flags))

            # The policy's k assignment does not depend on the split, so the score
            # and bucket of every GT row are resolved once.
            kidx = np.array([k_of_tok.get(t, MAX_K) for t in gt_tok], dtype=np.int32)
            sc = np.full(n_rows, np.inf)
            bk = np.zeros(n_rows, dtype=np.int32)
            for k in KS:
                sel = kidx == k
                sc[sel] = gt_scores[k][sel]
                bk[sel] = gt_bucket[k][sel]

            cov, radii, areas = [], [], []
            for si, (cal, test) in enumerate(splits):
                q = qs[si]
                test_m = np.isin(table["scene_idx"], test)
                qq = q[kidx, bk]
                covered = np.isfinite(sc) & (sc <= qq)
                cov.append(covered[test_m].mean())
                radii.append(np.nanmean(np.where(np.isinf(qq[test_m]), np.nan,
                                                 qq[test_m])))

                if do_area and si < N_AREA_SPLITS:
                    # Area is accumulated per (k, ego bucket) group, each with ITS
                    # OWN calibrated radius. Collapsing to one radius per k would
                    # erase exactly the adaptivity the ego policy is meant to
                    # exploit, and the comparison would be rigged against it.
                    test_scenes = set(test.tolist())
                    groups = []
                    for k in KS:
                        for b in range(n_buckets):
                            allowed = {t for t in tokens
                                       if k_of_tok.get(t, MAX_K) == k
                                       and tok_bucket[(k, t)] == b
                                       and scene_list.index(tok2scene[t]) in test_scenes}
                            if not allowed or not np.isfinite(q[k, b]):
                                continue
                            qcls = np.full(len(CLASS_NAMES), np.inf)
                            qcls[ids[0]] = q[k, b]
                            a, _ = space_metrics(preds[k], gt, ego_xy, k, BUDGET,
                                                 qcls, allowed)
                            groups.append((len(allowed), a))
                    if groups:
                        w = np.array([n for n, _ in groups], dtype=float)
                        v = np.array([a for _, a in groups])
                        areas.append(float((w * v).sum() / w.sum()))

            per_seed.append((rate, cov, radii, areas))
        rate = float(np.mean([ps[0] for ps in per_seed]))
        cov = [np.mean(ps[1]) for ps in per_seed]
        radii = [np.nanmean(ps[2]) for ps in per_seed]
        # Per-split areas averaged over seeds, kept per split for paired tests.
        area_splits = (np.mean([ps[3] for ps in per_seed], axis=0).tolist()
                       if per_seed[0][3] else [])
        areas = area_splits

        rows.append({
            "policy": f"{kind}({param})", "kind": kind, "param": float(param),
            "refresh_rate": rate,
            "coverage": float(np.mean(cov)),
            "mean_radius": float(np.nanmean(radii)),
            "area": float(np.mean(areas)) if areas else float("nan"),
            "area_splits": [float(a) for a in areas],
            "n_seeds": len(per_seed),
        })
        print(f"  {rows[-1]['policy']:<14} rate={rate:.3f} "
              f"cov={rows[-1]['coverage']:.3f} r={rows[-1]['mean_radius']:.2f}m "
              f"area={rows[-1]['area']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    _report(rows)
    with open(out_json, "w") as f:
        json.dump({"classes": list(CLASSES), "budget": BUDGET, "alpha": ALPHA,
                   "ego_bins": [float(x) for x in EGO_BINS],
                   "q_median": q_med.tolist(), "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out_json}")


def _report(rows):
    """Cost per forward pass spent: the curve the systems claim rests on."""
    print("\n" + "=" * 92)
    print(f"REFRESH POLICY — coverage pinned at {1 - ALPHA:.2f} by calibration, "
          f"so the cost is SPACE and COMPUTE")
    print("A policy is better if it holds a tighter bound for the same refresh "
          "rate.")
    print("=" * 92)
    print(f"{'policy':<16} {'refresh rate':>13} {'coverage':>9} "
          f"{'mean radius':>12} {'claimed area':>13}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: (-x["refresh_rate"], x["kind"])):
        print(f"{r['policy']:<16} {r['refresh_rate']:>13.3f} "
              f"{r['coverage']:>9.3f} {r['mean_radius']:>12.2f} "
              f"{r['area']:>13.4f}")
    print("-" * 92)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-area", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "refresh_policy.json"))
    args = ap.parse_args()
    run(args.out, not args.skip_area)


if __name__ == "__main__":
    main()
