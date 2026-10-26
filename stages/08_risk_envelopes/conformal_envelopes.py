"""Calibrated, shape-adaptive envelopes for stale detections (zero GPU).

[envelope_coverage.py](envelope_coverage.py) swept a global radius and read the
coverage off the curve. That is a hyperparameter chosen after seeing the answer,
and it says nothing about a scene the calibration never saw. This script replaces
the sweep with **split-conformal calibration** and then asks the design question
the calibration makes well-posed.

The reframe that makes everything else fall out:

    conformal PINS the coverage (that is what the guarantee buys), so coverage
    stops being the figure of merit and AREA becomes the free variable. A better
    envelope is one that holds the same coverage inside less claimed space.

So every design choice below competes on area at fixed coverage, not on coverage.

WHAT IS CALIBRATED. For each detection i (propagated by its own predicted
velocity, per the stage-08 result that the velocity estimate is not the
bottleneck) we attach an envelope whose size is `q * sigma_i` where

    sigma_i = 1 + beta * |v_i| * k * dt          (metres, per-DETECTION)

`sigma_i` uses the PREDICTED speed of the detection, never GT — otherwise the
method would not be deployable. `beta` is how much the envelope is allowed to
grow for fast objects; `beta = 0` is a single global radius. Conformal fixes the
overall level `q`, so beta only controls the RELATIVE allocation of space.

THE NONCONFORMITY SCORE. A GT object g is covered iff some same-class envelope
contains it, so the score is the smallest normalised distance over detections:

    s_g = min_i  ||x_g - c_i||_(rho) / sigma_i

and coverage at level q is exactly `s_g <= q`. Calibrating the quantile of s on
held-out scenes therefore calibrates the envelope set directly, isotropic or not.

ANISOTROPY. `||.||_(rho)` measures distance in the frame of the detection's own
propagation direction, stretched by an AREA-NEUTRAL aspect ratio rho:

    ||d||_(rho) = sqrt( (d_along / rho)^2 + (d_cross * rho)^2 )

An ellipse with semi-axes (rho*q*sigma, q*sigma/rho) has area pi*(q*sigma)^2 —
identical to the disc it replaces. So rho reshapes the claimed space without
being handed any extra of it, and any area win at fixed coverage is real. Static
detections (|v| ~ 0) have no meaningful direction and stay isotropic.

THE HONEST CAVEATS, stated up front because they bound what the guarantee means:

  - Split conformal guarantees marginal coverage under EXCHANGEABILITY. nuScenes
    objects are re-observed at 2 Hz within a scene, so scores are heavily
    correlated and exchangeability is violated. We therefore split by SCENE
    (never within one), repeat over many random splits, and REPORT the achieved
    test coverage against nominal rather than asserting the guarantee holds. The
    gap between the two is a measurement, and it is in the output table.
  - Marginal coverage can hide conditional failure. A radius that covers 90 % of
    all objects may cover 60 % of the fast ones. The per-speed breakdown is the
    diagnostic, and it is the thing `beta` exists to fix.
    MEASURED (d1_key, alpha=0.1, held-out): it does not need fixing. At
    beta = 0 coverage is 0.893 / 0.892 / 0.904 / 0.888 across the four speed
    strata — no conditional deficit exists, because class-conditional
    calibration already conditions on motion implicitly (cars and buses move,
    cones and barriers do not). Raising beta over-covers fast objects at the
    expense of static ones and costs area. `beta` is kept in the grid as the
    control that establishes this, not as a recommended setting.
  - A GT object with no same-class detection anywhere has s = inf and can never
    be covered. That imposes a hard FLOOR on achievable coverage, reported first:
    below it, the calibrated radius is correctly infinite.

Usage:

    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/08_risk_envelopes/conformal_envelopes.py
    python stages/08_risk_envelopes/conformal_envelopes.py --config d2_key
"""

import argparse
import json
import os
import time

import numpy as np

from envelope_coverage import (CLASS_NAMES, CONFIGS, MOVING_SPEED, NUSC_DT,
                               OUT_DIR, PRIMARY_BUDGET, SPEED_BINS,
                               build_gt_table, eligible_mask, load_ground_truth,
                               load_nusc, load_predictions)
from eval_subset import frame_index_in_scene, load_infos, token_to_scene

# ── Config ────────────────────────────────────────────────────────────────

# Speed-adaptive scale: sigma_i = 1 + BETA * |v_i| * k * dt, metres.
# 0.0 = one global radius for every detection (pure marginal conformal).
BETAS = (0.0, 0.5)

# Area-neutral aspect ratios for the along/cross-track ellipse. 1.0 = disc.
RHOS = (1.0, 1.25, 1.5, 2.0, 3.0)

ALPHAS = (0.3, 0.2, 0.1, 0.05)      # miss rate; nominal coverage = 1 - alpha
PRIMARY_ALPHA = 0.1

N_SPLITS = 40
N_CAL_SCENES = 12                   # of 23; the rest are test
RNG_SEED = 0

# A detection slower than this has no reliable heading, so its envelope stays
# isotropic no matter what rho is.
MIN_SPEED_FOR_HEADING = 0.5

# Claimed-area raster (as in envelope_coverage.py, but per-detection ellipses).
AREA_RADIUS = 50.0
AREA_CELL = 0.5
AREA_STRIDE = 4                     # every Nth frame; the statistic is a mean
N_AREA_SPLITS = 8                   # calibration draws the area is averaged over

# Well-populated, motion-relevant class used for the headline area comparison
# (index into CLASS_NAMES).
CAR_CLASS = 0


# ── Scores ────────────────────────────────────────────────────────────────

def _heading_frame(velo):
    """Unit along-track and cross-track vectors per detection.

    Returns:
        (u, w, has_heading) each (P, 2), (P, 2), (P,) bool. Rows without a
        reliable heading get u = (1, 0), w = (0, 1); callers must treat them as
        isotropic, which _stretched_norm does via `has_heading`.
    """
    speed = np.linalg.norm(velo, axis=1)
    has = speed >= MIN_SPEED_FOR_HEADING
    u = np.zeros_like(velo)
    u[:, 0] = 1.0
    u[has] = velo[has] / speed[has, None]
    w = np.stack([-u[:, 1], u[:, 0]], axis=1)
    return u, w, has


def _stretched_norm(delta, u, w, has_heading, rho):
    """Anisotropic distance in each detection's own propagation frame.

    Args:
        delta: (G, P, 2) GT centre minus envelope centre.
        u, w: (P, 2) along-track / cross-track unit vectors.
        has_heading: (P,) rows that may be stretched; the rest stay isotropic.
        rho: area-neutral aspect ratio (1.0 = disc).

    Returns:
        (G, P) distance whose unit ball is an ellipse of area pi (i.e. equal to
        the unit disc), elongated by rho along the direction of travel.
    """
    if rho == 1.0:
        return np.linalg.norm(delta, axis=2)
    d_along = np.einsum("gpc,pc->gp", delta, u)
    d_cross = np.einsum("gpc,pc->gp", delta, w)
    stretched = np.sqrt((d_along / rho) ** 2 + (d_cross * rho) ** 2)
    iso = np.linalg.norm(delta, axis=2)
    return np.where(has_heading[None, :], stretched, iso)


def nonconformity_scores(gt, pred, spans, n_rows, k, beta, rho, budget):
    """Per-GT score s_g = min over same-class detections of ||.||_(rho) / sigma.

    Coverage at level q is exactly `s <= q`, for any beta and rho — which is what
    lets one calibration routine serve every envelope design.

    Returns:
        (G,) float64; inf where the frame has no same-class detection.
    """
    out = np.full(n_rows, np.inf)
    for tok, g in gt.items():
        lo, hi = spans[tok]
        if hi == lo:
            continue
        p = pred.get(tok)
        if p is None or len(p["xy"]) == 0:
            continue
        p = {key: val[:budget] for key, val in p.items()}

        centres = p["xy"] + p["velo"] * (k * NUSC_DT) if k else p["xy"]
        delta = g["xy"][:, None, :] - centres[None, :, :]        # (G, P, 2)
        u, w, has = _heading_frame(p["velo"])
        dist = _stretched_norm(delta, u, w, has, rho)

        # sigma is a per-DETECTION scale from its own predicted speed, so the
        # envelope is computable at inference time without any GT.
        sigma = 1.0 + beta * np.linalg.norm(p["velo"], axis=1) * (k * NUSC_DT)
        score = dist / sigma[None, :]

        same = g["cls"][:, None] == p["cls"][None, :]
        out[lo:hi] = np.where(same, score, np.inf).min(axis=1)
    return out


# ── Split conformal ───────────────────────────────────────────────────────

def conformal_quantile(cal_scores, alpha):
    """Split-conformal threshold q with finite-sample coverage >= 1 - alpha.

    The standard ceil((n+1)(1-alpha))/n order statistic. Returns inf when the
    requested level exceeds what n calibration points can certify, or when more
    than alpha of them are already inf (undetected objects) — in both cases no
    finite radius achieves the target, which is the honest answer.
    """
    s = np.asarray(cal_scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return np.inf
    # Taken as an explicit order statistic rather than np.quantile, which
    # interpolates on (n-1)*q and lands one rank high — conservative, but then
    # the reported radius would not be the one the guarantee refers to.
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return np.inf
    return float(np.partition(s, rank - 1)[rank - 1])


def calibrate_and_test(scores, cls, scene_idx, cal_scenes, test_scenes, alpha):
    """Class-conditional calibration on cal scenes, evaluation on test scenes.

    Class-conditional (a separate q per class) rather than one pooled q, because
    the classes differ by an order of magnitude in how well they are localised
    under staleness.

    Returns:
        (q_per_class, covered, test_mask) where covered is a (G,) bool over all
        rows (meaningless outside test_mask) and q may contain inf.
    """
    cal_mask = np.isin(scene_idx, cal_scenes)
    test_mask = np.isin(scene_idx, test_scenes)

    q = np.full(len(CLASS_NAMES), np.inf)
    for c in range(len(CLASS_NAMES)):
        sel = cal_mask & (cls == c)
        if sel.sum() == 0:
            continue
        q[c] = conformal_quantile(scores[sel], alpha)

    # An object no detection reached has score inf and can NEVER be covered —
    # including when q itself is inf, where `inf <= inf` would otherwise score it
    # as a success and silently inflate coverage above the floor.
    covered = np.isfinite(scores) & (scores <= q[cls])
    return q, covered, test_mask


def make_splits(n_scenes, n_splits=N_SPLITS, n_cal=N_CAL_SCENES, seed=RNG_SEED):
    """Scene-disjoint calibration/test partitions, generated ONCE.

    Every design (beta, rho) must be scored on the identical partitions, or the
    comparison between them carries split-to-split variance and the paired area
    numbers mean nothing. Drawing them from a shared rng inside the design loop
    silently breaks that, so the splits are built here and passed in.
    """
    rng = np.random.default_rng(seed)
    return [(p[:n_cal], p[n_cal:])
            for p in (rng.permutation(n_scenes) for _ in range(n_splits))]


def run_splits(scores, table, splits, alpha):
    """Evaluate one design over the shared calibrate/test partitions.

    Two coverage numbers per split. `cov` counts a test object as covered when
    its class radius is inf (an unbounded envelope trivially contains it);
    `cov_bounded` counts it as NOT covered, i.e. coverage by finite envelopes
    only. The difference is the share of test objects in classes for which
    calibration produced no bound at all — those also cost zero claimed area,
    since an infinite radius cannot be rasterised, so `cov` alone flatters.

    Returns:
        dict with mean/sd achieved coverage (all, moving, bounded-only), the
        fraction of splits that reached nominal, the per-class q of every split,
        its median, the number of splits with q = inf per class and the median
        over the finite splits only, and held-out coverage per GT speed stratum.
    """
    cls, scene_idx = table["cls"], table["scene_idx"]
    moving = table["speed"] >= MOVING_SPEED

    cov_all, cov_mov, cov_bnd, unbounded, qs = [], [], [], [], []
    by_speed = [[] for _ in SPEED_BINS]
    by_class = [[] for _ in CLASS_NAMES]   # held-out coverage where q is FINITE
    for cal, test in splits:
        q, covered, tmask = calibrate_and_test(scores, cls, scene_idx, cal,
                                               test, alpha)
        bounded = np.isfinite(q[cls])
        if tmask.sum():
            cov_all.append(covered[tmask].mean())
            cov_bnd.append((covered & bounded)[tmask].mean())
            unbounded.append((~bounded)[tmask].mean())
        m = tmask & moving
        if m.sum():
            cov_mov.append(covered[m].mean())
        for c in range(len(CLASS_NAMES)):
            cm = tmask & (cls == c)
            if cm.sum() and np.isfinite(q[c]):
                by_class[c].append(covered[cm].mean())
        for b, (lo, hi) in enumerate(SPEED_BINS):
            sm = tmask & (table["speed"] >= lo) & (table["speed"] < hi)
            if sm.sum():
                by_speed[b].append(covered[sm].mean())
        qs.append(q)

    cov_all, cov_mov, cov_bnd = np.array(cov_all), np.array(cov_mov), np.array(cov_bnd)
    q_arr = np.array(qs)                                  # (n_splits, n_classes)
    finite = np.isfinite(q_arr)
    q_med_finite = [float(np.median(q_arr[finite[:, c], c])) if finite[:, c].any()
                    else float("inf") for c in range(q_arr.shape[1])]
    return {
        "cov_mean": float(cov_all.mean()), "cov_sd": float(cov_all.std()),
        "cov_bounded_mean": float(cov_bnd.mean()),
        "frac_test_unbounded": float(np.mean(unbounded)),
        "cov_moving_mean": float(cov_mov.mean()) if len(cov_mov) else float("nan"),
        "frac_splits_at_nominal": float((cov_all >= 1 - alpha).mean()),
        # Median over ALL splits (inf counts as a value, so a class that is inf in
        # half the splits reports inf or a huge number), plus the two numbers that
        # say what that median actually means.
        "q_median": np.median(q_arr, axis=0).tolist(),
        "q_n_inf": (~finite).sum(axis=0).astype(int).tolist(),
        "q_median_finite": q_med_finite,
        "cov_by_speed_heldout": [float(np.mean(v)) if v else float("nan")
                                 for v in by_speed],
        # Does a finite radius transfer to unseen scenes? Mean over the splits
        # where the class got a finite q and has test objects.
        "cov_by_class_heldout_finite": [float(np.mean(v)) if v else float("nan")
                                        for v in by_class],
        # Every split's own q, so claimed area can be evaluated under the SAME
        # calibration draw that produced the coverage rather than at a summary
        # statistic — per-class order statistics jump between splits, and a
        # single median q makes the area column pure jitter.
        "q_splits": [qq.tolist() for qq in qs],
    }


# ── Claimed area ──────────────────────────────────────────────────────────

def claimed_area(pred, ego_xy, k, beta, rho, q_per_class, budget,
                 stride=AREA_STRIDE, classes=None, tokens=None):
    """Mean fraction of the 50 m ego disc claimed by the union of envelopes.

    Rasterises per detection (ellipses have per-detection size and orientation,
    so the KD-tree shortcut used for equal discs no longer applies). Only every
    `stride`-th frame is rasterised; the statistic is a mean over frames.

    Args:
        classes: optional iterable of class ids to restrict the union to. The
            aggregate union is dominated by whichever class got the largest
            calibrated radius, so a single-class area is the only way to see what
            beta and rho do to the envelope shape itself.
        tokens: set of sample tokens to average over — the split's TEST scenes,
            eligible frames only. Pricing the calibration scenes too would let
            every split share most of its frames, which is what made the
            per-split area comparison pseudo-replicated.
    """
    g = np.arange(-AREA_RADIUS, AREA_RADIUS + 1e-9, AREA_CELL)
    gx, gy = np.meshgrid(g, g, indexing="xy")
    inside = (gx ** 2 + gy ** 2) <= AREA_RADIUS ** 2
    cell_x, cell_y = gx[inside], gy[inside]
    n_cells = cell_x.size

    items = [(t, p) for t, p in pred.items()
             if t in ego_xy and (tokens is None or t in tokens)]
    tot, n = 0.0, 0
    for i, (tok, p) in enumerate(items):
        if i % stride:
            continue
        p = {key: val[:budget] for key, val in p.items()}
        if len(p["xy"]) == 0:
            n += 1
            continue
        if classes is not None:
            sel = np.isin(p["cls"], list(classes))
            p = {key: val[sel] for key, val in p.items()}
            if len(p["xy"]) == 0:
                n += 1
                continue
        centres = p["xy"] + p["velo"] * (k * NUSC_DT) if k else p["xy"]
        local = centres - ego_xy[tok]
        sigma = 1.0 + beta * np.linalg.norm(p["velo"], axis=1) * (k * NUSC_DT)
        r = q_per_class[p["cls"]] * sigma                      # (P,) metres
        u, w, has = _heading_frame(p["velo"])

        claimed = np.zeros(n_cells, dtype=bool)
        for j in range(len(local)):
            if not np.isfinite(r[j]) or r[j] <= 0:
                continue
            dx = cell_x - local[j, 0]
            dy = cell_y - local[j, 1]
            # Cheap bounding-box reject before the ellipse test.
            reach = r[j] * (rho if has[j] else 1.0)
            near = (np.abs(dx) <= reach) & (np.abs(dy) <= reach)
            if not near.any():
                continue
            ddx, ddy = dx[near], dy[near]
            if rho == 1.0 or not has[j]:
                hit = (ddx ** 2 + ddy ** 2) <= r[j] ** 2
            else:
                a = ddx * u[j, 0] + ddy * u[j, 1]
                c = ddx * w[j, 0] + ddy * w[j, 1]
                hit = ((a / rho) ** 2 + (c * rho) ** 2) <= r[j] ** 2
            idx = np.flatnonzero(near)[hit]
            claimed[idx] = True
        tot += claimed.mean()
        n += 1
    return tot / max(n, 1)


# ── Reporting ─────────────────────────────────────────────────────────────

def print_floor(scores_by_k, table, configs):
    """Fraction of GT no envelope can ever contain — the hard coverage ceiling."""
    print("\n" + "=" * 92)
    print("COVERAGE FLOOR — GT objects with NO same-class detection at any "
          "distance")
    print("Below 1 - floor, the calibrated radius is correctly infinite.")
    print("=" * 92)
    print(f"{'config':<10} {'n_gt':>8} {'unreachable':>13} {'max coverage':>14} "
          f"{'moving unreach.':>17}")
    print("-" * 92)
    for c in configs:
        s = scores_by_k[c]
        mask = eligible_mask(table, CONFIGS[c][1])
        mov = mask & (table["speed"] >= MOVING_SPEED)
        bad = ~np.isfinite(s[mask])
        bad_m = ~np.isfinite(s[mov])
        print(f"{c:<10} {int(mask.sum()):>8} {bad.mean():>13.4f} "
              f"{1 - bad.mean():>14.4f} {bad_m.mean():>17.4f}")
    print("-" * 92)


def print_alpha_sweep(rows, config_key):
    """Nominal vs achieved coverage — does the guarantee survive 2 Hz scenes?"""
    print("\n" + "=" * 92)
    print(f"SPLIT-CONFORMAL CALIBRATION — {config_key}, class-conditional, "
          f"{N_SPLITS} scene-disjoint splits ({N_CAL_SCENES} cal / "
          f"{23 - N_CAL_SCENES} test)")
    print("Exchangeability is violated within scenes, so ACHIEVED is measured, "
          "not assumed.")
    print("=" * 92)
    print(f"{'beta':>6} {'alpha':>7} {'nominal':>9} {'achieved':>10} {'sd':>7} "
          f"{'splits>=nom':>12} {'moving cov':>11}")
    print("-" * 92)
    for r in rows:
        print(f"{r['beta']:>6.2f} {r['alpha']:>7.2f} {1 - r['alpha']:>9.3f} "
              f"{r['cov_mean']:>10.3f} {r['cov_sd']:>7.3f} "
              f"{r['frac_splits_at_nominal']:>12.2f} "
              f"{r['cov_moving_mean']:>11.3f}")
    print("-" * 92)


def print_design_table(rows, config_key, alpha):
    """Area at FIXED coverage — the comparison conformal makes possible.

    Two area columns. The aggregate union is dominated by whichever class drew
    the largest calibrated radius (the rare, badly-localised ones), so it is
    reported but not used to decide anything. `car` is the well-powered,
    motion-relevant class where the envelope shape is actually observable.
    """
    print("\n" + "=" * 92)
    print(f"DESIGN COMPARISON — {config_key}, alpha={alpha} "
          f"(coverage pinned by calibration, so AREA is the figure of merit)")
    print(f"Area averaged over {N_AREA_SPLITS} calibration splits, paired "
          f"across designs; +- is the split sd.")
    print("=" * 92)
    print(f"{'beta':>6} {'rho':>6} {'cov':>7} {'moving':>8} "
          f"{'area all':>16} {'area car':>16} {'car vs disc':>12}")
    print("-" * 92)
    base = next((r["area_car"] for r in rows
                 if r["beta"] == 0.0 and r["rho"] == 1.0), None)
    for r in rows:
        rel = (f"{r['area_car'] / base:>12.3f}"
               if base and np.isfinite(base) else f"{'--':>12}")
        print(f"{r['beta']:>6.2f} {r['rho']:>6.2f} {r['cov_mean']:>7.3f} "
              f"{r['cov_moving_mean']:>8.3f} "
              f"{r['area']:>9.4f}+-{r['area_sd']:<5.4f} "
              f"{r['area_car']:>9.4f}+-{r['area_car_sd']:<5.4f} {rel}")
    print("-" * 92)


def print_rho_paired(rows, beta=0.0):
    """Paired per-split area difference vs the disc — settles whether rho helps.

    The across-split spread of the area (~10 % of its value) is larger than the
    differences between shapes, so the marginal means in the design table cannot
    decide this. Every design saw the SAME calibration splits and the SAME
    detections — only q and the shape differ — so the per-split difference is
    tightly paired and is the statistic with the power to answer it.
    """
    base = next((r for r in rows if r["beta"] == beta and r["rho"] == 1.0), None)
    if base is None or "area_car_splits" not in base:
        return
    b = np.asarray(base["area_car_splits"])
    print("\n" + "=" * 92)
    print(f"ANISOTROPY, PAIRED — car claimed area vs the disc (beta={beta}), "
          f"per calibration split")
    print("Equal-area ellipse by construction, so any reduction is shape, "
          "not budget.")
    print("Splits overlap (resamples of 23 scenes): the sign test is a consistency "
          "check, not significance.")
    print("=" * 92)
    print(f"{'rho':>6} {'mean area':>11} {'mean delta':>12} {'rel':>8} "
          f"{'splits better':>15} {'sign p':>9}")
    print("-" * 92)
    for r in rows:
        if r["beta"] != beta or "area_car_splits" not in r:
            continue
        a = np.asarray(r["area_car_splits"])
        d = a - b
        n, wins = len(d), int((d < 0).sum())
        if r["rho"] == 1.0:      # the baseline against itself: d == 0 exactly,
            print(f"{r['rho']:>6.2f} {a.mean():>11.4f} {'--':>12} "
                  f"{1.0:>8.3f} {'--':>8}{'':<6} {'--':>9}")
            continue             # a sign test on it would report a fake p.
        # Exact two-sided binomial sign test; with n=8 the floor is p=0.0078.
        # The splits are overlapping resamples of the same 23 scenes, so this is
        # a consistency check across splits, not an independent-samples test.
        from scipy import stats
        p = float(stats.binomtest(wins, n, 0.5).pvalue) if n else float("nan")
        print(f"{r['rho']:>6.2f} {a.mean():>11.4f} {d.mean():>+12.5f} "
              f"{a.mean() / b.mean():>8.3f} {wins:>8}/{n:<6} {p:>9.4f}")
    print("-" * 92)


def print_calibrated_radii(rows):
    """Median calibrated radius per class — shows what drives the aggregate area.

    A class whose detections are poorly localised under staleness draws a huge
    q, and since every one of its detections then claims that radius, a handful
    of them dominate the union. This table is why the aggregate area column
    above is not the figure of merit.
    """
    row = next((r for r in rows if r["beta"] == 0.0 and r["rho"] == 1.0), None)
    if row is None:
        return
    print("\n" + "=" * 92)
    print("CALIBRATED RADIUS per class (beta=0, rho=1, median over splits)")
    print("=" * 92)
    print("A median over splits that are partly inf is not a radius: read it with "
          "the inf count.")
    print(f"{'class':<24} {'q median':>10} {'inf splits':>11} {'q median (finite)':>18} "
          f"{'held-out cov (finite q)':>24}")
    print("-" * 92)
    n_sp = len(row["q_splits"])
    rows_ = zip(CLASS_NAMES, row["q_median"], row["q_n_inf"], row["q_median_finite"],
                row["cov_by_class_heldout_finite"])
    for name, q, ninf, qf, hc in sorted(rows_, key=lambda x: (-x[2], -x[3])):
        print(f"{name:<24} {q:>10.2f} {ninf:>5}/{n_sp:<5} {qf:>18.2f} {hc:>24.3f}")
    print(f"test objects in classes with q = inf: {row['frac_test_unbounded']:.3f} "
          f"of the test set; coverage counting them as uncovered: "
          f"{row['cov_bounded_mean']:.3f} (vs {row['cov_mean']:.3f})")
    print("-" * 92)


def print_conditional(table, config_key, rows_by_beta, alpha):
    """Per-speed HELD-OUT coverage under a marginally-calibrated radius.

    This is where marginal conformal is allowed to fail, and the reason `beta`
    exists: a single radius calibrated over a mostly-static population will
    under-cover exactly the objects that matter. Coverage is measured on each
    split's test scenes with that split's own q and averaged over splits (an
    earlier version applied the median q to all scenes, calibration included).
    """
    k = CONFIGS[config_key][1]
    mask = eligible_mask(table, k)
    print("\n" + "=" * 92)
    print(f"CONDITIONAL COVERAGE by GT speed — {config_key}, alpha={alpha}")
    print(f"Marginal target is {1 - alpha:.2f} everywhere; gaps here are "
          f"conditional failures.")
    print("=" * 92)
    header = f"{'speed (m/s)':<16} {'n':>7}"
    for beta in rows_by_beta:
        header += f" {'beta=' + format(beta, '.2f'):>12}"
    print(header)
    print("-" * 92)
    for b, (lo, hi) in enumerate(SPEED_BINS):
        m = mask & (table["speed"] >= lo) & (table["speed"] < hi)
        label = f"[{lo:.1f}, {'inf' if not np.isfinite(hi) else f'{hi:.1f}'})"
        line = f"{label:<16} {int(m.sum()):>7}"
        for beta, r in rows_by_beta.items():
            line += f" {r['cov_by_speed_heldout'][b]:>12.3f}"
        print(line)
    print("-" * 92)


# ── Main ──────────────────────────────────────────────────────────────────

def run(config_key, do_area, out_json):
    """Calibrate, sweep the design space, and price each design by claimed area."""
    nusc, cfg = load_nusc()
    by_scene = load_infos()
    tok2scene = token_to_scene(by_scene)
    fidx = frame_index_in_scene(by_scene)
    ego_xy = {i["token"]: np.asarray(i["ego2global_translation"][:2], np.float64)
              for fs in by_scene.values() for i in fs}

    ref = load_predictions(nusc, cfg, "fresh")
    tokens = sorted(ref.keys())
    gt = load_ground_truth(nusc, cfg, tokens)
    table, spans, scenes = build_gt_table(gt, tok2scene, fidx)
    n_rows, n_scenes = len(table["cls"]), len(scenes)
    print(f"[load] {n_rows} GT objects, {n_scenes} scenes")

    k = CONFIGS[config_key][1]
    pred = load_predictions(nusc, cfg, config_key, tokens)
    scene_of = {s: i for i, s in enumerate(scenes)}
    # Per-split area is priced on that split's TEST scenes, eligible frames only.
    eligible_tok = [t for t in tokens if fidx.get(t, -1) >= k and t in tok2scene]

    # Restrict every statistic to frames that actually got k frames of staleness.
    keep = eligible_mask(table, k)

    # ── Scores for the whole design grid ──
    print(f"\n[scores] {len(BETAS)} betas x {len(RHOS)} rhos ...", flush=True)
    scores = {}
    t0 = time.time()
    for beta in BETAS:
        for rho in RHOS:
            scores[(beta, rho)] = nonconformity_scores(
                gt, pred, spans, n_rows, k, beta, rho, PRIMARY_BUDGET)
    print(f"[scores] done in {time.time() - t0:.0f}s")

    floor_scores = {config_key: scores[(0.0, 1.0)]}
    print_floor(floor_scores, table, [config_key])

    sub = {kk: vv[keep] for kk, vv in table.items()}
    sub_scores = {kk: v[keep] for kk, v in scores.items()}

    # One set of partitions for every design and every alpha, so all rows below
    # are paired and differences are attributable to the design alone.
    splits = make_splits(n_scenes)

    # ── alpha sweep at rho = 1 (calibration behaviour) ──
    alpha_rows = []
    for beta in BETAS:
        for alpha in ALPHAS:
            r = run_splits(sub_scores[(beta, 1.0)], sub, splits, alpha)
            alpha_rows.append({"beta": beta, "rho": 1.0, "alpha": alpha, **r})
    print_alpha_sweep(alpha_rows, config_key)

    # ── design grid at the primary alpha, priced by claimed area ──
    design_rows = []
    for beta in BETAS:
        for rho in RHOS:
            r = run_splits(sub_scores[(beta, rho)], sub, splits, PRIMARY_ALPHA)
            row = {"beta": beta, "rho": rho, "alpha": PRIMARY_ALPHA, **r}
            row.update(area=float("nan"), area_sd=float("nan"),
                       area_car=float("nan"), area_car_sd=float("nan"))
            if do_area:
                t1 = time.time()
                # Paired over splits: every design is priced under the SAME
                # calibration draws, so the comparison is not decided by which
                # split each design happened to get.
                qs = [np.asarray(q) for q in r["q_splits"][:N_AREA_SPLITS]]
                test_tok = [{t for t in eligible_tok
                             if scene_of[tok2scene[t]] in set(test.tolist())}
                            for _, test in splits[:N_AREA_SPLITS]]
                agg = [claimed_area(pred, ego_xy, k, beta, rho, q,
                                    PRIMARY_BUDGET, tokens=tt)
                       for q, tt in zip(qs, test_tok)]
                car = [claimed_area(pred, ego_xy, k, beta, rho, q,
                                    PRIMARY_BUDGET, classes=[CAR_CLASS], tokens=tt)
                       for q, tt in zip(qs, test_tok)]
                # Keep every split's area, not just the mean: the rho effect is
                # smaller than the across-split spread, so only the PAIRED
                # per-split differences can decide it.
                row.update(area=float(np.mean(agg)), area_sd=float(np.std(agg)),
                           area_car=float(np.mean(car)),
                           area_car_sd=float(np.std(car)),
                           area_splits=[float(x) for x in agg],
                           area_car_splits=[float(x) for x in car])
                print(f"  area beta={beta} rho={rho}: all "
                      f"{row['area']:.4f}+-{row['area_sd']:.4f}  car "
                      f"{row['area_car']:.4f}+-{row['area_car_sd']:.4f} "
                      f"({time.time() - t1:.0f}s)", flush=True)
            design_rows.append(row)
    print_design_table(design_rows, config_key, PRIMARY_ALPHA)
    print_rho_paired(design_rows, beta=0.0)
    print_calibrated_radii(design_rows)

    # ── conditional coverage, held out ──
    rows_by_beta = {beta: run_splits(sub_scores[(beta, 1.0)], sub, splits, PRIMARY_ALPHA)
                    for beta in BETAS}
    print_conditional(table, config_key, rows_by_beta, PRIMARY_ALPHA)

    payload = {
        "config": config_key, "k": k, "budget": PRIMARY_BUDGET,
        "n_splits": N_SPLITS, "n_cal_scenes": N_CAL_SCENES,
        "class_names": list(CLASS_NAMES),
        "floor": float((~np.isfinite(scores[(0.0, 1.0)][keep])).mean()),
        "alpha_sweep": alpha_rows,
        "design_grid": design_rows,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="d1_key",
                    choices=[c for c in CONFIGS if CONFIGS[c][1] > 0])
    ap.add_argument("--skip-area", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(OUT_DIR, f"conformal_{args.config}.json")
    run(args.config, not args.skip_area, out)


if __name__ == "__main__":
    main()
