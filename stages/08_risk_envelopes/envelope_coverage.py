"""Do stale detections MISS objects, or merely MISLOCATE them? (zero GPU)

Every mitigation tried so far (stages 04/07) was scored with mAP, which matches a
detection to GT only within 0.5-4 m. That is a point-estimate contract: "you are
here". This script asks the weaker, safety-shaped question instead:

    Given a detection computed from photons that are k frames old, can we bound
    where the object is NOW, and how much space does that bound cost?

Two things are measured, in one pass over detections that already exist on disk.

  (1) THE GATE — missed or mislocated?
      Recall of GT at generous BEV centre distances (0.5 ... 10 m) with NO
      propagation, at a matched emission budget, stale vs fresh on the same
      tokens. Inflation can only rescue objects that were DETECTED and put in the
      wrong place; it cannot rescue objects the detector never fired on. So if
      recall@10m holds up while recall@2m collapses, the envelope axis is alive.
      If recall@10m collapses too, objects are being LOST and no radius helps.

  (2) THE CURVE — coverage vs cost.
      Propagate each detection by k*dt (velocity variants below), then sweep an
      envelope radius r and report the fraction of GT objects whose centre falls
      inside some same-class envelope. Coverage alone is trivially winnable
      (r -> inf covers everything), so it is always reported against the cost:
      the fraction of the 50 m ego disc claimed as possibly-occupied by the union
      of envelopes. The fresh detector's own curve is the ceiling.

Three propagation variants, because the velocity term is the part most likely to
be noise (stage 04 Phase 2 found velocity extrapolation contributed nothing to
memory reference points, and `vel_warp` was later shown to be inside scene
variance):

    static       no propagation at all                         <- the null
    velo         + predicted velocity * k * dt                 <- the method
    oracle_velo  + velocity of the nearest same-class GT       <- upper bound

`oracle_velo` is a diagnostic, not a deployable method: it reads GT at time t to
pick a velocity, and for a fast object the nearest GT may be a different
instance. It exists only to separate "the velocity ESTIMATE is bad" from "the
envelope FRAMING is bad".

Note what is NOT re-derived here: `vel_warp` (stage 04) edited
`memory_reference_point` after `post_update_memory`, i.e. it changed what the
NEXT frame reads and never moved the boxes emitted at frame t. This script
operates on the emitted detections, which is a different site (output, not
state).

Correctness gates, printed at the top of every run:
  - GT is loaded and filtered with the devkit's own `add_center_dist` +
    `filter_eval_boxes` (class range, num_pts > 0), so the object set is exactly
    the one the published mAP numbers were computed against.
  - The first k frames of every scene are excluded for a delay=k config: the
    FeatureSubstitutor clamps the delay to the cache length there, so those
    frames are effectively fresh. The fresh reference is restricted to the same
    tokens whenever the two are compared.

Usage:

    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/08_risk_envelopes/envelope_coverage.py
    python stages/08_risk_envelopes/envelope_coverage.py --configs fresh d1_key \
        --skip-area
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# eval_subset / bootstrap_ci live in stage 04 and are the shared primitives for
# every zero-GPU re-analysis in this project (cached devkit GT, scene/frame
# bookkeeping, cluster bootstrap). Import them rather than re-implement.
_STAGE04 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "04_streampetr_latent"))
sys.path.insert(0, _STAGE04)

from bootstrap_ci import bootstrap_mean_ci, paired_tests  # noqa: E402
from eval_subset import (DATAROOT, EVAL_SET, EVAL_VERSION, NUSC_DT,  # noqa: E402
                         VERSION, config_results_path, frame_index_in_scene,
                         load_infos, token_to_scene)

# ── Config ────────────────────────────────────────────────────────────────

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")

# name -> (dump dir under stage 04 dumps/, staleness in frames).
# 'd1_naive' is kept alongside 'd1_key' so the ordering "fix the input geometry
# first, then bound the residual" can be read straight off the curve.
CONFIGS = {
    "fresh":    ("delay0",         0),
    "d1_naive": ("delay1",         1),
    "d1_key":   ("honest_d1_key",  1),
    "d2_key":   ("honest_d2_key",  2),
    "d3_key":   ("honest_d3_key",  3),
}

MAX_BOXES_PER_SAMPLE = 500          # devkit cap; StreamPETR emits 300

# The 10 detection classes of detection_cvpr_2019, in the devkit's own order.
# Only used to give each class a stable integer id for masking.
CLASS_NAMES = ("car", "truck", "bus", "trailer", "construction_vehicle",
               "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")
CLASS_INDEX = {n: i for i, n in enumerate(CLASS_NAMES)}

# Emission budget: top-N detections per frame by score. A fixed budget (rather
# than a fixed score threshold) keeps the comparison controlled, because
# staleness itself shifts the score distribution.
BUDGETS = (30, 100, 300)
PRIMARY_BUDGET = 100

GATE_DISTS = (0.5, 1.0, 2.0, 4.0, 10.0)
RADII = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
VARIANTS = ("static", "velo", "oracle_velo")

# GT speed strata (m/s). The claim the envelope has to support is about objects
# that MOVE; static objects are already handled by the honest key-side lift.
SPEED_BINS = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, np.inf))
MOVING_SPEED = 2.0                  # threshold for the headline "moving" subset

ORACLE_MATCH_R = 10.0               # gate for oracle-velocity GT association

# Cost denominator: fraction of the 50 m ego-centred disc claimed by the union
# of envelopes. 50 m is the devkit's largest class range.
AREA_RADIUS = 50.0
AREA_CELL = 0.5                     # metres per grid cell

HEADLINE_RADII = (2.0, 4.0, 6.0, 10.0)


# ── Loading ───────────────────────────────────────────────────────────────

def load_nusc():
    """Load NuScenes metadata + the detection eval config (~30 s, once)."""
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory

    print(f"[load] NuScenes {VERSION} metadata from {DATAROOT} ...", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    cfg = config_factory(EVAL_VERSION)
    return nusc, cfg


def _boxes_to_arrays(eval_boxes, tokens):
    """Convert devkit EvalBoxes to per-token numpy arrays.

    Args:
        eval_boxes: devkit EvalBoxes (already centre-dist annotated + filtered).
        tokens: tokens to keep.

    Returns:
        dict token -> dict with
            xy    (P, 2) float64 global BEV centre
            velo  (P, 2) float64 global velocity, m/s (nan -> 0)
            cls   (P,)   int32 class index into CLASS_NAMES
            score (P,)   float64 (GT boxes get 1.0)
        Rows are sorted by descending score, so a top-N budget is a prefix.
    """
    out = {}
    for tok in tokens:
        boxes = eval_boxes.boxes.get(tok, [])
        if not boxes:
            out[tok] = {"xy": np.zeros((0, 2)), "velo": np.zeros((0, 2)),
                        "cls": np.zeros(0, np.int32), "score": np.zeros(0)}
            continue
        xy = np.array([b.translation[:2] for b in boxes], dtype=np.float64)
        velo = np.nan_to_num(
            np.array([b.velocity[:2] for b in boxes], dtype=np.float64))
        # GT boxes carry detection_score = -1.0; their order is irrelevant since
        # the top-N budget only ever truncates predictions.
        score = np.array([float(b.detection_score) for b in boxes],
                         dtype=np.float64)
        cls = np.array([CLASS_INDEX[b.detection_name] for b in boxes],
                       dtype=np.int32)

        order = np.argsort(-score)
        out[tok] = {"xy": xy[order], "velo": velo[order],
                    "cls": cls[order], "score": score[order]}
    return out


def load_predictions(nusc, cfg, config_key, tokens=None):
    """Load one config's saved detections, devkit-filtered.

    Applies exactly the devkit's own prediction pipeline (`add_center_dist` then
    `filter_eval_boxes`, i.e. the per-class range filter), so the box set matches
    the one behind that config's recorded mAP.
    """
    from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                              load_prediction)
    from nuscenes.eval.detection.data_classes import DetectionBox

    path = config_results_path(CONFIGS[config_key][0])
    boxes, _ = load_prediction(path, MAX_BOXES_PER_SAMPLE, DetectionBox,
                               verbose=False)
    boxes = add_center_dist(nusc, boxes)
    boxes = filter_eval_boxes(nusc, boxes, cfg.class_range, verbose=False)
    toks = list(boxes.sample_tokens) if tokens is None else list(tokens)
    return _boxes_to_arrays(boxes, toks)


def load_ground_truth(nusc, cfg, tokens):
    """Load GT for `tokens`, filtered by class range and num_pts > 0."""
    from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                              load_gt)
    from nuscenes.eval.detection.data_classes import DetectionBox

    print("[load] devkit GT for the val split (~30 s) ...", flush=True)
    gt = load_gt(nusc, EVAL_SET, DetectionBox, verbose=False)
    gt = add_center_dist(nusc, gt)
    gt = filter_eval_boxes(nusc, gt, cfg.class_range, verbose=False)
    return _boxes_to_arrays(gt, tokens)


# ── GT table ──────────────────────────────────────────────────────────────

def build_gt_table(gt, tok2scene, fidx):
    """Flatten per-token GT into one aligned row table.

    Every downstream statistic is a masked aggregation over these rows, so the
    per-config token eligibility (first k frames excluded) and the speed/class
    strata are all just boolean masks — no re-matching.

    Returns:
        (table, spans) where table is a dict of (G,) arrays
            cls, speed, frame_idx, scene_idx
        and spans maps token -> (start, stop) row range.
    """
    cls, speed, frame, scene, spans = [], [], [], [], {}
    scenes = sorted(set(tok2scene.values()))
    scene_index = {s: i for i, s in enumerate(scenes)}

    start = 0
    for tok, g in gt.items():
        n = len(g["cls"])
        spans[tok] = (start, start + n)
        start += n
        if n == 0:
            continue
        cls.append(g["cls"])
        speed.append(np.linalg.norm(g["velo"], axis=1))
        frame.append(np.full(n, fidx.get(tok, -1), dtype=np.int32))
        scene.append(np.full(n, scene_index[tok2scene[tok]], dtype=np.int32))

    table = {
        "cls": np.concatenate(cls) if cls else np.zeros(0, np.int32),
        "speed": np.concatenate(speed) if speed else np.zeros(0),
        "frame_idx": np.concatenate(frame) if frame else np.zeros(0, np.int32),
        "scene_idx": np.concatenate(scene) if scene else np.zeros(0, np.int32),
    }
    return table, spans, scenes


# ── Matching ──────────────────────────────────────────────────────────────

def _oracle_velocity(pred, gt_tok):
    """Velocity of the nearest same-class GT within ORACLE_MATCH_R, per pred.

    Falls back to the predicted velocity where no GT is close enough. This reads
    GT and is therefore a diagnostic upper bound only (see module docstring).
    """
    if len(pred["xy"]) == 0 or len(gt_tok["xy"]) == 0:
        return pred["velo"]
    d = np.linalg.norm(pred["xy"][:, None, :] - gt_tok["xy"][None, :, :], axis=2)
    same = pred["cls"][:, None] == gt_tok["cls"][None, :]
    d = np.where(same, d, np.inf)
    j = np.argmin(d, axis=1)
    ok = d[np.arange(len(j)), j] <= ORACLE_MATCH_R
    velo = pred["velo"].copy()
    velo[ok] = gt_tok["velo"][j[ok]]
    return velo


def propagated_centres(pred, gt_tok, k, variant):
    """Envelope centres at time t for detections computed from t-k photons.

    Args:
        pred: per-token prediction arrays (already budget-truncated).
        gt_tok: per-token GT arrays (only read by the oracle variant).
        k: staleness in frames.
        variant: one of VARIANTS.

    Returns:
        (P, 2) float64 propagated BEV centres.
    """
    if k == 0 or variant == "static":
        return pred["xy"]
    velo = pred["velo"] if variant == "velo" else _oracle_velocity(pred, gt_tok)
    return pred["xy"] + velo * (k * NUSC_DT)


def min_distance_per_gt(gt, pred, spans, n_rows, k, variant, budget):
    """Distance from every GT object to the nearest same-class envelope centre.

    This is the one expensive step, and everything else (gate table, coverage
    curves, per-class and per-speed breakdowns) is a threshold on its output —
    coverage at radius r is simply `dmin <= r`.

    Returns:
        (G,) float64, inf where the token has no same-class detection.
    """
    dmin = np.full(n_rows, np.inf)
    for tok, g in gt.items():
        lo, hi = spans[tok]
        if hi == lo:
            continue
        p = pred.get(tok)
        if p is None or len(p["xy"]) == 0:
            continue
        p = {key: val[:budget] for key, val in p.items()}
        centres = propagated_centres(p, g, k, variant)
        d = np.linalg.norm(g["xy"][:, None, :] - centres[None, :, :], axis=2)
        same = g["cls"][:, None] == p["cls"][None, :]
        d = np.where(same, d, np.inf)
        dmin[lo:hi] = d.min(axis=1)
    return dmin


# ── Cost: claimed area ────────────────────────────────────────────────────

def claimed_area_fractions(pred, gt, ego_xy, k, variant, budget, radii,
                           log_every=200, tokens=None):
    """Fraction of the 50 m ego disc claimed by the union of envelopes.

    Coverage is meaningless without this: r -> inf covers every object and
    paralyses any consumer. Computed by rasterising the ego disc at AREA_CELL
    resolution and taking, per cell, the distance to the nearest envelope centre
    (one KD-tree query per frame serves every radius).

    Args:
        tokens: optional set of sample tokens to average over. Pass the stale
            config's eligible tokens (frame index >= k) for BOTH the stale and the
            fresh config, so the scene-start frames that fall back to fresh are
            excluded and the two areas are measured on the same frames.

    Returns:
        (len(radii),) float64 mean claimed fraction over tokens.
    """
    from scipy.spatial import cKDTree

    g = np.arange(-AREA_RADIUS, AREA_RADIUS + 1e-9, AREA_CELL)
    gx, gy = np.meshgrid(g, g, indexing="xy")
    inside = (gx ** 2 + gy ** 2) <= AREA_RADIUS ** 2
    grid = np.stack([gx[inside], gy[inside]], axis=1)   # (C, 2) ego-local

    acc = np.zeros(len(radii))
    n = 0
    t0 = time.time()
    for i, (tok, p) in enumerate(pred.items()):
        if tok not in ego_xy or (tokens is not None and tok not in tokens):
            continue
        p = {key: val[:budget] for key, val in p.items()}
        if len(p["xy"]) == 0:
            n += 1
            continue
        centres = propagated_centres(p, gt.get(tok, {"xy": np.zeros((0, 2)),
                                                     "cls": np.zeros(0, np.int32),
                                                     "velo": np.zeros((0, 2))}),
                                     k, variant)
        local = centres - ego_xy[tok]
        d, _ = cKDTree(local).query(grid, k=1)
        acc += [(d <= r).mean() for r in radii]
        n += 1
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    area {i + 1}/{len(pred)} frames "
                  f"({rate:.1f}/s)", flush=True)
    return acc / max(n, 1)


# ── Aggregation ───────────────────────────────────────────────────────────

def coverage(dmin, mask, radius):
    """Fraction of the masked GT rows within `radius` of an envelope centre."""
    if mask.sum() == 0:
        return float("nan")
    return float((dmin[mask] <= radius).mean())


def per_scene_coverage(dmin, mask, scene_idx, n_scenes, radius):
    """Coverage computed separately per scene (the independent unit, n=23)."""
    out = np.full(n_scenes, np.nan)
    for s in range(n_scenes):
        m = mask & (scene_idx == s)
        if m.sum():
            out[s] = (dmin[m] <= radius).mean()
    return out


def eligible_mask(table, k):
    """GT rows whose frame actually received k frames of staleness."""
    return table["frame_idx"] >= k


# ── Reporting ─────────────────────────────────────────────────────────────

def print_gate(results, table, budget):
    """THE GATE: recall vs distance, no propagation. Missed or mislocated?"""
    print("\n" + "=" * 92)
    print(f"GATE — recall of GT at generous centre distance, NO propagation "
          f"(top-{budget} boxes/frame)")
    print("Inflation can rescue MISLOCATED objects, never MISSED ones.")
    print("Each stale config is compared to fresh on ITS OWN eligible tokens.")
    print("=" * 92)
    head = f"{'config':<10} {'n_gt':>7}"
    for d in GATE_DISTS:
        head += f" {'@' + format(d, '.1f') + 'm':>9}"
    print(head)
    print("-" * 92)
    for name, r in results.items():
        k = CONFIGS[name][1]
        mask = eligible_mask(table, k)
        d = r["dmin"]["static"][budget]
        line = f"{name:<10} {int(mask.sum()):>7}"
        for dist in GATE_DISTS:
            line += f" {coverage(d, mask, dist):>9.3f}"
        print(line)
        if k > 0:
            base = results["fresh"]["dmin"]["static"][budget]
            line = f"{'  (fresh)':<10} {'':>7}"
            for dist in GATE_DISTS:
                line += f" {coverage(base, mask, dist):>9.3f}"
            print(line + "   <- same tokens")
    print("-" * 92)


def print_gate_per_class(results, table, budget, config_key, near, far):
    """Per-class near vs far recall — where the losses are misses vs shifts."""
    print("\n" + "=" * 92)
    print(f"GATE per class — {config_key} vs fresh, recall@{near}m and @{far}m")
    print("Large @far with small @near = mislocated (envelope helps).")
    print("Small @far = genuinely missed (envelope cannot help).")
    print("=" * 92)
    k = CONFIGS[config_key][1]
    mask = eligible_mask(table, k)
    d_st = results[config_key]["dmin"]["static"][budget]
    d_fr = results["fresh"]["dmin"]["static"][budget]
    print(f"{'class':<22} {'n':>6} {'fresh@' + str(near):>10} "
          f"{'stale@' + str(near):>10} {'fresh@' + str(far):>10} "
          f"{'stale@' + str(far):>10} {'far ret':>9}")
    print("-" * 92)
    for ci, cname in enumerate(CLASS_NAMES):
        m = mask & (table["cls"] == ci)
        if m.sum() == 0:
            continue
        fn, sn = coverage(d_fr, m, near), coverage(d_st, m, near)
        ff, sf = coverage(d_fr, m, far), coverage(d_st, m, far)
        ret = sf / ff if ff > 0 else float("nan")
        print(f"{cname:<22} {int(m.sum()):>6} {fn:>10.3f} {sn:>10.3f} "
              f"{ff:>10.3f} {sf:>10.3f} {ret:>9.1%}")
    print("-" * 92)


def print_coverage_curve(results, table, budget, config_key):
    """Coverage vs radius for each propagation variant, incl. the null."""
    k = CONFIGS[config_key][1]
    mask = eligible_mask(table, k)
    moving = mask & (table["speed"] >= MOVING_SPEED)
    print("\n" + "=" * 92)
    print(f"COVERAGE vs ENVELOPE RADIUS — {config_key} (k={k}), "
          f"top-{budget} boxes/frame")
    print("'static' is the null: if velo does not beat it, propagation is noise.")
    print("=" * 92)
    print(f"{'':>7} {'--- all GT objects ---':^46} | "
          f"{'--- moving GT only (>= ' + str(MOVING_SPEED) + ' m/s) ---':^44}")
    print(f"{'r (m)':>7} " + "".join(f"{v:>14}" for v in VARIANTS)
          + f" {'fresh':>10} | " + "".join(f"{v:>14}" for v in VARIANTS))
    print("-" * 92)
    fresh_d = results["fresh"]["dmin"]["static"][budget]
    for r in RADII:
        line = f"{r:>7.1f} "
        for v in VARIANTS:
            line += f"{coverage(results[config_key]['dmin'][v][budget], mask, r):>14.3f}"
        line += f" {coverage(fresh_d, mask, r):>10.3f} |"
        for v in VARIANTS:
            line += f"{coverage(results[config_key]['dmin'][v][budget], moving, r):>14.3f}"
        print(line)
    print("-" * 92)


def print_speed_breakdown(results, table, budget, config_key, variant, radii):
    """Coverage by GT speed — the envelope's whole point is moving objects."""
    k = CONFIGS[config_key][1]
    mask = eligible_mask(table, k)
    print("\n" + "=" * 92)
    print(f"COVERAGE by GT speed — {config_key}, variant='{variant}', "
          f"top-{budget}")
    print("=" * 92)
    print(f"{'speed (m/s)':<16} {'n':>7}" + "".join(
        f" {'r=' + format(r, '.0f'):>9}" for r in radii)
        + "".join(f" {'fresh r=' + format(r, '.0f'):>13}" for r in radii))
    print("-" * 92)
    d = results[config_key]["dmin"][variant][budget]
    d_fr = results["fresh"]["dmin"]["static"][budget]
    for lo, hi in SPEED_BINS:
        m = mask & (table["speed"] >= lo) & (table["speed"] < hi)
        label = f"[{lo:.1f}, {'inf' if not np.isfinite(hi) else f'{hi:.1f}'})"
        line = f"{label:<16} {int(m.sum()):>7}"
        line += "".join(f" {coverage(d, m, r):>9.3f}" for r in radii)
        line += "".join(f" {coverage(d_fr, m, r):>13.3f}" for r in radii)
        print(line)
    print("-" * 92)


def print_cost(area, config_key, variant, budget):
    """Coverage is free unless the claimed area is reported next to it."""
    print("\n" + "=" * 92)
    print(f"COST — fraction of the {AREA_RADIUS:.0f} m ego disc claimed as "
          f"possibly-occupied ({config_key}, '{variant}', top-{budget})")
    print("=" * 92)
    print(f"{'r (m)':>7} {'claimed area frac':>20}")
    print("-" * 92)
    for r, a in zip(RADII, area):
        print(f"{r:>7.1f} {a:>20.4f}")
    print("-" * 92)


def _paired_block(a_dmin, b_dmin, mask, table, n_scenes, title, col_a, col_b):
    """Paired per-scene coverage comparison of two dmin arrays at HEADLINE_RADII."""
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(f"{'r (m)':>7} {col_a:>11} {col_b:>11} {'mean d':>9} "
          f"{'95% CI on delta':>22} {'wins':>8} {'sign p':>9}")
    print("-" * 92)
    out = {}
    for r in HEADLINE_RADII:
        a = per_scene_coverage(a_dmin, mask, table["scene_idx"], n_scenes, r)
        b = per_scene_coverage(b_dmin, mask, table["scene_idx"], n_scenes, r)
        t = paired_tests(a, b)
        am, _, _ = bootstrap_mean_ci(a)
        bm, _, _ = bootstrap_mean_ci(b)
        print(f"{r:>7.1f} {am:>11.3f} {bm:>11.3f} {t['mean_delta']:>+9.3f} "
              f"[{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}] "
              f"{t['wins']:>4}/{t['n_scenes']:<3} {t['sign_p']:>9.3f}")
        out[str(r)] = {"a_mean": am, "b_mean": bm, **t}
    print("-" * 92)
    return out


def print_stats(results, table, scenes, budget, config_key, variant):
    """Scene-level paired tests — the bar vel_warp failed to clear.

    Two comparisons, and the second is the one that decides whether propagation
    is a method or a coin flip:
      (a) stale+propagation vs FRESH   — how much coverage is still missing.
      (b) stale+propagation vs STATIC  — does the velocity term do anything?
    """
    k = CONFIGS[config_key][1]
    mask = eligible_mask(table, k) & (table["speed"] >= MOVING_SPEED)
    n = len(scenes)
    d_var = results[config_key]["dmin"][variant][budget]
    d_null = results[config_key]["dmin"]["static"][budget]
    d_fresh = results["fresh"]["dmin"]["static"][budget]

    vs_fresh = _paired_block(
        d_var, d_fresh, mask, table, n,
        f"SCENE-LEVEL (n={n}) — coverage of MOVING objects (>= {MOVING_SPEED} m/s)\n"
        f"(a) {config_key} '{variant}' vs FRESH — the residual gap",
        "stale cov", "fresh cov")
    vs_null = _paired_block(
        d_var, d_null, mask, table, n,
        f"(b) {config_key} '{variant}' vs '{'static'}' — is propagation real?\n"
        f"This is the test vel_warp failed (13/23 scenes, p=0.68).",
        f"{variant} cov", "static cov")
    return {"vs_fresh": vs_fresh, "vs_static": vs_null}


def print_equivalent_radius(results, table, budget, config_key, variant):
    """Radius the stale system needs to match fresh coverage at a given radius.

    This is the price of the envelope stated the way a consumer would ask it:
    'the fresh detector bounds this object within r0 m; running k frames late,
    how big must the bound be for the same coverage?'
    """
    k = CONFIGS[config_key][1]
    mask = eligible_mask(table, k) & (table["speed"] >= MOVING_SPEED)
    d_var = results[config_key]["dmin"][variant][budget]
    d_fresh = results["fresh"]["dmin"]["static"][budget]

    fine = np.arange(0.0, 40.01, 0.1)
    stale_cov = np.array([coverage(d_var, mask, r) for r in fine])

    print("\n" + "=" * 92)
    print(f"EQUIVALENT RADIUS — {config_key} '{variant}', moving objects "
          f"(>= {MOVING_SPEED} m/s)")
    print("What the envelope costs to buy back fresh-level coverage.")
    print("=" * 92)
    print(f"{'fresh r0 (m)':>13} {'fresh cov':>11} {'stale r (m)':>13} "
          f"{'radius x':>10} {'area x':>10}")
    print("-" * 92)
    out = {}
    for r0 in HEADLINE_RADII:
        target = coverage(d_fresh, mask, r0)
        hit = np.where(stale_cov >= target)[0]
        if len(hit) == 0:
            print(f"{r0:>13.1f} {target:>11.3f} {'unreachable':>13} "
                  f"{'--':>10} {'--':>10}")
            out[str(r0)] = {"fresh_cov": target, "stale_radius": None}
            continue
        r_eq = float(fine[hit[0]])
        print(f"{r0:>13.1f} {target:>11.3f} {r_eq:>13.1f} "
              f"{r_eq / r0:>10.1f} {(r_eq / r0) ** 2:>10.1f}")
        out[str(r0)] = {"fresh_cov": target, "stale_radius": r_eq,
                        "radius_ratio": r_eq / r0}
    print("-" * 92)
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def run(config_keys, budgets, skip_area, out_json):
    """Load every config, match, and print the gate, curves, cost and stats."""
    nusc, cfg = load_nusc()
    assert set(cfg.class_range) == set(CLASS_NAMES), \
        "devkit class set differs from CLASS_NAMES"

    by_scene = load_infos()
    tok2scene = token_to_scene(by_scene)
    fidx = frame_index_in_scene(by_scene)
    ego_xy = {i["token"]: np.asarray(i["ego2global_translation"][:2], np.float64)
              for fs in by_scene.values() for i in fs}

    # Token set = whatever the fresh run scored, which is the 914-sample
    # partial-blob val subset every other number in this project uses.
    ref = load_predictions(nusc, cfg, "fresh")
    tokens = sorted(ref.keys())
    print(f"[load] {len(tokens)} sample tokens, {len(by_scene)} scenes")

    gt = load_ground_truth(nusc, cfg, tokens)
    table, spans, scenes = build_gt_table(gt, tok2scene, fidx)
    n_rows = len(table["cls"])
    print(f"[load] {n_rows} GT objects after devkit filtering "
          f"(class range + num_pts > 0)")
    print(f"[load] GT speed: median {np.nanmedian(table['speed']):.2f} m/s, "
          f"moving (>= {MOVING_SPEED} m/s): "
          f"{int((table['speed'] >= MOVING_SPEED).sum())} objects")

    results = {}
    for name in config_keys:
        dump, k = CONFIGS[name]
        print(f"\n[match] {name} (dump={dump}, k={k}) ...", flush=True)
        pred = ref if name == "fresh" else load_predictions(nusc, cfg, name,
                                                            tokens)
        n_box = np.mean([len(p["xy"]) for p in pred.values()])
        print(f"  {n_box:.1f} boxes/frame after filtering")
        dmin = {}
        for v in VARIANTS:
            if k == 0 and v != "static":
                dmin[v] = dmin["static"]          # propagation is a no-op
                continue
            dmin[v] = {}
            for b in budgets:
                t0 = time.time()
                dmin[v][b] = min_distance_per_gt(gt, pred, spans, n_rows, k, v, b)
                print(f"  variant={v:<12} budget={b:<4} "
                      f"{time.time() - t0:.1f}s", flush=True)
        results[name] = {"dmin": dmin, "pred": pred, "k": k}

    # ── The two questions ──
    print_gate(results, table, PRIMARY_BUDGET)
    stale_keys = [c for c in config_keys if CONFIGS[c][1] > 0]
    primary = "d1_key" if "d1_key" in stale_keys else (stale_keys or ["fresh"])[0]
    print_gate_per_class(results, table, PRIMARY_BUDGET, primary, 2.0, 10.0)

    for c in stale_keys:
        print_coverage_curve(results, table, PRIMARY_BUDGET, c)
    print_speed_breakdown(results, table, PRIMARY_BUDGET, primary, "velo",
                          HEADLINE_RADII)

    area = {}
    # Both areas on the primary config's eligible frames (first k frames of each
    # scene fall back to fresh features and are excluded everywhere else too).
    k_primary = CONFIGS[primary][1]
    area_tokens = {t for t in results[primary]["pred"] if fidx.get(t, -1) >= k_primary}
    if not skip_area:
        for c in [primary, "fresh"]:
            if c not in results:
                continue
            print(f"\n[area] rasterising claimed area for {c} ...", flush=True)
            a = claimed_area_fractions(results[c]["pred"], gt, ego_xy,
                                       results[c]["k"],
                                       "velo" if results[c]["k"] else "static",
                                       PRIMARY_BUDGET, RADII, tokens=area_tokens)
            area[c] = a.tolist()
            print_cost(a, c, "velo" if results[c]["k"] else "static",
                       PRIMARY_BUDGET)

    equiv = print_equivalent_radius(results, table, PRIMARY_BUDGET, primary, "velo")
    stats = print_stats(results, table, scenes, PRIMARY_BUDGET, primary, "velo")

    # ── Persist everything the tables were derived from ──
    payload = {
        "configs": {c: {"dump": CONFIGS[c][0], "k": CONFIGS[c][1]}
                    for c in config_keys},
        "budget": PRIMARY_BUDGET,
        "radii": list(RADII),
        "gate_dists": list(GATE_DISTS),
        "n_gt": int(n_rows),
        "gate": {c: {str(d): coverage(results[c]["dmin"]["static"][PRIMARY_BUDGET],
                                      eligible_mask(table, CONFIGS[c][1]), d)
                     for d in GATE_DISTS} for c in config_keys},
        "curves": {
            c: {v: {str(r): coverage(results[c]["dmin"][v][PRIMARY_BUDGET],
                                     eligible_mask(table, CONFIGS[c][1]), r)
                    for r in RADII} for v in VARIANTS}
            for c in config_keys},
        "curves_moving": {
            c: {v: {str(r): coverage(
                results[c]["dmin"][v][PRIMARY_BUDGET],
                eligible_mask(table, CONFIGS[c][1]) & (table["speed"] >= MOVING_SPEED),
                r) for r in RADII} for v in VARIANTS}
            for c in config_keys},
        "area_fraction": area,
        "equivalent_radius": equiv,
        "scene_stats": stats,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--skip-area", action="store_true",
                    help="skip the KD-tree claimed-area rasterisation (~1 min)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "envelope_coverage.json"))
    args = ap.parse_args()

    budgets = sorted(set(args.budgets) | {PRIMARY_BUDGET})
    configs = list(dict.fromkeys(["fresh"] + args.configs))
    run(configs, budgets, args.skip_area, args.out)


if __name__ == "__main__":
    main()
