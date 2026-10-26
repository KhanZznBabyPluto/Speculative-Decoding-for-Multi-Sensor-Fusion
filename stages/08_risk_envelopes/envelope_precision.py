"""How much of the claimed space is justified? (zero GPU)

Coverage says the true object is inside some envelope. It says nothing about the
other envelopes, and there are ~100 of them per frame against ~50 real objects.
A consumer that is handed 100 "the object might be here" regions, most of which
are about nothing, has not been helped. This script measures the other half.

THE KNOB THAT ACTUALLY TRADES THESE OFF is the emission budget, not the radius.
Radius is no longer free — conformal calibration fixes it once coverage is
chosen. What remains free is HOW MANY detections are allowed to emit an
envelope, and it cuts both ways:

    fewer envelopes -> fewer false ones, but each surviving object must be
                       covered by a sparser set, so the calibrated radius GROWS
    more envelopes  -> smaller radius, but more claimed space is about nothing

So the experiment sweeps the budget, re-calibrates the radius at each budget to
hold coverage at 1 - alpha, and asks which operating point claims the least
space. That is the question the refresh policy will need answered first.

FOUR MEASURES, reported per budget on held-out scenes:

  coverage      fraction of GT inside some same-class envelope (pinned by
                calibration; reported as a check that it really is pinned)
  precision     fraction of EMITTED envelopes containing >= 1 same-class GT.
                The direct analogue of detection precision for a set-valued
                output.
  redundancy    envelopes per covered GT object. 10 envelopes on one car means
                9 of them are spending space to say something already said.
  claimed       fraction of the 50 m ego disc claimed as possibly-occupied.
                The honest reference is the SAME sweep run on the fresh
                detector (--config fresh): the ratio is the space that running
                late actually costs. Absolute claimed area is not interpretable
                on its own.

`wasted` (claimed area further than JUSTIFIED_R from any annotated object) is
computed and reported but MUST NOT be read as waste. With a calibrated radius of
~4 m and a 2 m core, a perfect envelope is already ~77 % "wasted" by this
definition — it penalises a bound for being a bound, and it saturates at
0.94-0.96 for every budget. It is kept only as a crude free-space indicator.

Aggregate numbers over all 10 classes are dominated by whichever class drew the
largest calibrated radius (bus ~39 m, bicycle ~33 m on this data), so a handful
of their detections cover most of the disc. Use --classes to restrict the whole
pipeline — calibration, precision and area — to an interpretable class.

Usage:
    conda activate /ibex/user/khangea/conda-environments/streampetr_env
    python stages/08_risk_envelopes/envelope_precision.py
"""

import argparse
import json
import os
import time

import numpy as np

from conformal_envelopes import (AREA_CELL, AREA_RADIUS, AREA_STRIDE,
                                 CLASS_NAMES, MIN_SPEED_FOR_HEADING,
                                 _heading_frame, _stretched_norm,
                                 conformal_quantile, make_splits)
from envelope_coverage import (CONFIGS, NUSC_DT, OUT_DIR, build_gt_table,
                               eligible_mask, load_ground_truth, load_nusc,
                               load_predictions)
from eval_subset import frame_index_in_scene, load_infos, token_to_scene

# ── Config ────────────────────────────────────────────────────────────────

# Emission budgets: top-N detections per frame by score.
BUDGETS = (10, 20, 30, 50, 100, 200, 300)

ALPHA = 0.1
BETA = 0.0      # falsified as a fix in the previous session; kept at the null
RHO = 1.25      # the paired winner: -4.5 % area at equal coverage, 8/8 splits

N_SPLITS = 40
N_AREA_SPLITS = 4               # rasterisation is the expensive part

# A cell is "justified" if some annotated object sits within this distance —
# roughly a car half-length, so the object's own footprint counts as justified.
JUSTIFIED_R = 2.0


# ── One pass: per-GT and per-detection scores ─────────────────────────────

def score_pass(gt, pred, spans, n_rows, k, budget):
    """Normalised distances from both sides, in a single sweep over frames.

    Coverage needs, per GT, the distance to the nearest envelope. Precision
    needs, per DETECTION, the distance to the nearest same-class GT. Both come
    off the same matrix, so they are computed together.

    Returns:
        gt_score:  (G,) per-GT min normalised distance (inf if none)
        det_score: (D,) per-detection min normalised distance (inf if none)
        det_cls:   (D,) class id per emitted detection
        det_scene: (D,) scene index per emitted detection
        det_tok:   (D,) token index per emitted detection
    """
    gt_score = np.full(n_rows, np.inf)
    d_score, d_cls, d_scene, d_tok = [], [], [], []

    for ti, (tok, g) in enumerate(gt.items()):
        p = pred.get(tok)
        if p is None or len(p["xy"]) == 0:
            continue
        p = {key: val[:budget] for key, val in p.items()}
        n_det = len(p["xy"])

        centres = p["xy"] + p["velo"] * (k * NUSC_DT) if k else p["xy"]
        sigma = 1.0 + BETA * np.linalg.norm(p["velo"], axis=1) * (k * NUSC_DT)
        u, w, has = _heading_frame(p["velo"])

        lo, hi = spans[tok]
        if hi > lo:
            delta = g["xy"][:, None, :] - centres[None, :, :]
            dist = _stretched_norm(delta, u, w, has, RHO) / sigma[None, :]
            same = g["cls"][:, None] == p["cls"][None, :]
            masked = np.where(same, dist, np.inf)
            gt_score[lo:hi] = masked.min(axis=1)
            det_min = masked.min(axis=0)              # (P,) nearest same-class GT
        else:
            det_min = np.full(n_det, np.inf)

        d_score.append(det_min)
        d_cls.append(p["cls"])
        d_scene.append(np.full(n_det, g["scene_idx"], dtype=np.int32))
        d_tok.append(np.full(n_det, ti, dtype=np.int32))

    cat = lambda xs, dt: (np.concatenate(xs) if xs else np.zeros(0, dt))
    return (gt_score, cat(d_score, np.float64), cat(d_cls, np.int32),
            cat(d_scene, np.int32), cat(d_tok, np.int32))


# ── Claimed and wasted space ──────────────────────────────────────────────

def space_metrics(pred, gt, ego_xy, k, budget, q_per_class, tokens_allowed,
                  stride=AREA_STRIDE):
    """Claimed and unjustified fractions of the ego disc, averaged over frames.

    Returns:
        (claimed, wasted) each a fraction of the 50 m disc. `wasted` is the part
        of the claimed region with no annotated object within JUSTIFIED_R.
    """
    from scipy.spatial import cKDTree

    g = np.arange(-AREA_RADIUS, AREA_RADIUS + 1e-9, AREA_CELL)
    gx, gy = np.meshgrid(g, g, indexing="xy")
    inside = (gx ** 2 + gy ** 2) <= AREA_RADIUS ** 2
    cx, cy = gx[inside], gy[inside]

    # Stride over the ALLOWED frames, not over all of `pred`: striding first and
    # filtering second can leave a small allowed set with zero sampled frames,
    # which then reported area 0 (and refresh_policy weighted it as real).
    items = [(t, p) for t, p in pred.items() if t in ego_xy and t in tokens_allowed]
    tot_claimed, tot_wasted, n = 0.0, 0.0, 0
    for i, (tok, p) in enumerate(items):
        if i % stride:
            continue
        n += 1
        p = {key: val[:budget] for key, val in p.items()}
        if len(p["xy"]) == 0:
            continue

        centres = p["xy"] + p["velo"] * (k * NUSC_DT) if k else p["xy"]
        local = centres - ego_xy[tok]
        sigma = 1.0 + BETA * np.linalg.norm(p["velo"], axis=1) * (k * NUSC_DT)
        r = q_per_class[p["cls"]] * sigma
        u, w, has = _heading_frame(p["velo"])

        claimed = np.zeros(cx.size, dtype=bool)
        for j in range(len(local)):
            if not np.isfinite(r[j]) or r[j] <= 0:
                continue
            dx, dy = cx - local[j, 0], cy - local[j, 1]
            reach = r[j] * (RHO if has[j] else 1.0)
            near = (np.abs(dx) <= reach) & (np.abs(dy) <= reach)
            if not near.any():
                continue
            ddx, ddy = dx[near], dy[near]
            if RHO == 1.0 or not has[j]:
                hit = (ddx ** 2 + ddy ** 2) <= r[j] ** 2
            else:
                a = ddx * u[j, 0] + ddy * u[j, 1]
                c = ddx * w[j, 0] + ddy * w[j, 1]
                hit = ((a / RHO) ** 2 + (c * RHO) ** 2) <= r[j] ** 2
            claimed[np.flatnonzero(near)[hit]] = True

        tot_claimed += claimed.mean()

        gt_xy = gt[tok]["xy"]
        if len(gt_xy) == 0:
            tot_wasted += claimed.mean()          # nothing here justifies anything
            continue
        d, _ = cKDTree(gt_xy - ego_xy[tok]).query(np.stack([cx, cy], 1), k=1)
        tot_wasted += (claimed & (d > JUSTIFIED_R)).mean()

    n = max(n, 1)
    return tot_claimed / n, tot_wasted / n


# ── Main ──────────────────────────────────────────────────────────────────

def _restrict_classes(boxes, class_ids):
    """Keep only the given class ids in a per-token array dict."""
    out = {}
    for tok, b in boxes.items():
        sel = np.isin(b["cls"], class_ids)
        out[tok] = {key: val[sel] for key, val in b.items()
                    if isinstance(val, np.ndarray)}
    return out


def run(config_key, do_area, out_json, class_names=None):
    """Sweep the emission budget, recalibrating the radius at each one."""
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

    k = CONFIGS[config_key][1]
    pred = load_predictions(nusc, cfg, config_key, tokens)

    if class_names:
        ids = [CLASS_NAMES.index(c) for c in class_names]
        # Restrict BEFORE building the GT table so calibration, precision and
        # area all see the same restricted population.
        gt = _restrict_classes(gt, ids)
        pred = _restrict_classes(pred, ids)
        table, spans, scenes = build_gt_table(gt, tok2scene, fidx)
        n_rows = len(table["cls"])
        print(f"[setup] restricted to {class_names}: {n_rows} GT objects")

    keep = eligible_mask(table, k)
    scene_of_tok = {t: sorted(set(tok2scene.values())).index(tok2scene[t])
                    for t in tokens}
    for tok in gt:
        gt[tok]["scene_idx"] = scene_of_tok[tok]
    eligible_tok = {t for t in tokens if fidx.get(t, -1) >= k}

    splits = make_splits(n_scenes, n_splits=N_SPLITS)
    print(f"[setup] {n_rows} GT ({int(keep.sum())} eligible), {n_scenes} scenes, "
          f"alpha={ALPHA}, beta={BETA}, rho={RHO}", flush=True)

    rows = []
    for budget in BUDGETS:
        t0 = time.time()
        gt_s, det_s, det_cls, det_scene, det_tok = score_pass(gt, pred, spans, n_rows,
                                                              k, budget)
        # Detections from the first k frames of a scene (fresh fallback) are not
        # stale and must not enter precision or envelopes-per-frame.
        gt_toks = list(gt.keys())
        det_ok = np.array([gt_toks[i] in eligible_tok for i in det_tok], dtype=bool)
        cov, prec, redun, n_env = [], [], [], []
        claimed, wasted = [], []
        for si, (cal, test) in enumerate(splits):
            q = np.full(len(CLASS_NAMES), np.inf)
            cal_m = np.isin(table["scene_idx"], cal) & keep
            for c in range(len(CLASS_NAMES)):
                sel = cal_m & (table["cls"] == c)
                if sel.sum():
                    q[c] = conformal_quantile(gt_s[sel], ALPHA)

            test_m = np.isin(table["scene_idx"], test) & keep
            test_set = set(test.tolist())
            n_frames = sum(1 for t in eligible_tok
                           if scene_of_tok[t] in test_set)
            covered = np.isfinite(gt_s) & (gt_s <= q[table["cls"]])
            cov.append(covered[test_m].mean())

            dm = np.isin(det_scene, test) & det_ok
            hits = np.isfinite(det_s) & (det_s <= q[det_cls])
            prec.append(hits[dm].mean() if dm.sum() else np.nan)
            n_env.append(dm.sum() / max(n_frames, 1))
            redun.append(hits[dm].sum() / max(covered[test_m].sum(), 1))

            if do_area and si < N_AREA_SPLITS:
                allowed = {t for t in eligible_tok if scene_of_tok[t] in set(test)}
                a, wst = space_metrics(pred, gt, ego_xy, k, budget, q, allowed)
                claimed.append(a)
                wasted.append(wst)

        row = {
            "budget": budget,
            "coverage": float(np.mean(cov)),
            "precision": float(np.nanmean(prec)),
            "redundancy": float(np.mean(redun)),
            "envelopes_per_frame": float(np.mean(n_env)),
            "claimed": float(np.mean(claimed)) if claimed else float("nan"),
            "claimed_sd": float(np.std(claimed)) if claimed else float("nan"),
            "wasted": float(np.mean(wasted)) if wasted else float("nan"),
            "q_car": float(np.median([
                conformal_quantile(gt_s[np.isin(table["scene_idx"], cal) & keep
                                        & (table["cls"] == 0)], ALPHA)
                for cal, _ in splits[:8]])),
        }
        rows.append(row)
        print(f"  budget={budget:<4} cov={row['coverage']:.3f} "
              f"prec={row['precision']:.3f} redun={row['redundancy']:.1f} "
              f"q_car={row['q_car']:.2f}m claimed={row['claimed']:.4f} "
              f"wasted={row['wasted']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    _report(rows)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"config": config_key, "k": k, "alpha": ALPHA, "beta": BETA,
                   "rho": RHO, "justified_r": JUSTIFIED_R, "rows": rows}, f,
                  indent=2)
    print(f"\nSaved -> {out_json}")


def _report(rows):
    """The operating-point table: what does each budget buy and cost?"""
    print("\n" + "=" * 92)
    print(f"EMISSION BUDGET SWEEP — coverage held at {1 - ALPHA:.2f} by "
          f"recalibrating the radius at every budget")
    print("Radius is not free any more, so the budget is the knob that trades "
          "precision against space.")
    print("=" * 92)
    print(f"{'budget':>7} {'env/frame':>10} {'coverage':>9} {'precision':>10} "
          f"{'redundancy':>11} {'car q (m)':>10} {'claimed':>9} {'wasted':>8} "
          f"{'wast/clm':>9}")
    print("-" * 92)
    for r in rows:
        frac = r["wasted"] / r["claimed"] if r["claimed"] else float("nan")
        print(f"{r['budget']:>7} {r['envelopes_per_frame']:>10.1f} "
              f"{r['coverage']:>9.3f} {r['precision']:>10.3f} "
              f"{r['redundancy']:>11.1f} {r['q_car']:>10.2f} "
              f"{r['claimed']:>9.4f} {r['wasted']:>8.4f} {frac:>9.3f}")
    print("-" * 92)
    best = min((r for r in rows if np.isfinite(r["claimed"])),
               key=lambda r: r["claimed"], default=None)
    if best:
        print(f"Least space claimed at the SAME coverage: budget="
              f"{best['budget']} ({best['claimed']:.4f} of the disc, "
              f"precision {best['precision']:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="d1_key")
    ap.add_argument("--skip-area", action="store_true")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="restrict the whole pipeline to these class names")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.config + ("_" + "_".join(args.classes) if args.classes else "")
    out = args.out or os.path.join(OUT_DIR, f"precision_{tag}.json")
    run(args.config, not args.skip_area, out, args.classes)


if __name__ == "__main__":
    main()
