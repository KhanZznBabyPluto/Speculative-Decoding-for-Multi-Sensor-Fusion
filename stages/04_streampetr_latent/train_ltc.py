"""Phase 4 — train LTC speculator on (stale, fresh) pairs from
`dump_ltc_pairs.py`.

Loss: MSE on `pred_emb` vs `fresh_new_emb` plus weighted MSE on
`pred_ref` vs `fresh_new_ref`. Ref points are stored in global frame
in the pairs; we center per-pair (subtract mean) to keep MLP inputs
~unit-scale and re-add at eval time if reporting in metres.

Train/val split: scene-level (a held-out fraction of scene_tokens
goes to val). Random k sampling per batch (uniform over the dump's
configured staleness values).

Pilot success criterion (per NOTES):
    val emb_l2 < something interesting, AND
    when the trained checkpoint is plugged into
    `measure_stale_map_mitigation.py --mitigation ltc`, the resulting
    mAP at delay=1 ≥ 0.20 (vs analytical vel_warp's 0.138).

Run:
    cd stages/04_streampetr_latent
    python train_ltc.py \
        --pairs dumps/ltc_pairs.pt \
        --steps 5000 \
        --val-frac 0.2 \
        --ckpt-out dumps/ltc_pilot.pt
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ltc import LTC, LTCConfig, NUSC_DT_SECONDS                          # noqa: E402

REF_CENTER_KEY = "_ref_center"  # internal book-keeping for centering


class LTCPairsDataset(Dataset):
    """Wraps a list of (stale, fresh) pair dicts loaded from
    dump_ltc_pairs.py. For each item we ego-center the ref points
    (subtract the mean of carried+stale refs) so the MLP inputs are
    ~unit-scale. We store the center back on the item so prediction
    metrics can be reported unscaled if needed."""

    def __init__(self, pairs: List[dict]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        carried_ref = p["carried_ref"].float()
        stale_ref = p["stale_new_ref"].float()
        fresh_ref = p["fresh_new_ref"].float()
        # Center using the carried block's xy median (robust against
        # outlier tokens). z left alone.
        center_xy = carried_ref[:, :2].median(dim=0).values              # [2]
        center = torch.tensor([center_xy[0], center_xy[1], 0.0])
        out = {
            "stale_new_emb":  p["stale_new_emb"].float(),
            "stale_new_ref":  stale_ref - center,
            "stale_new_velo": p["stale_new_velo"].float(),
            "carried_emb":    p["carried_emb"].float(),
            "carried_ref":    carried_ref - center,
            "fresh_new_emb":  p["fresh_new_emb"].float(),
            "fresh_new_ref":  fresh_ref - center,
            "k":              torch.tensor(p["k"], dtype=torch.long),
            "_ref_center":    center,
        }
        # v3 pairs include fresh_new_velo so the model can learn an
        # MLN-consistent triplet (emb + ref + velo). v2 dumps don't have
        # it; we keep this optional so old dumps still load (the v2-style
        # training path with predict_velo=False ignores the absence).
        if "fresh_new_velo" in p:
            out["fresh_new_velo"] = p["fresh_new_velo"].float()
        return out


def collate(batch):
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out


def scene_split(pairs: List[dict], val_frac: float, seed: int):
    """Disjoint scene-level split. Last `val_frac` of unique scene
    tokens (in stable load order) go to val."""
    scene_tokens = []
    seen = set()
    for p in pairs:
        t = p["scene_token"]
        if t not in seen:
            seen.add(t)
            scene_tokens.append(t)
    n_val = max(1, int(round(val_frac * len(scene_tokens))))
    val_set = set(scene_tokens[-n_val:])
    train_pairs = [p for p in pairs if p["scene_token"] not in val_set]
    val_pairs   = [p for p in pairs if p["scene_token"]     in val_set]
    return train_pairs, val_pairs, scene_tokens[-n_val:]


def per_token_l2(pred, target):
    return (pred - target).norm(dim=-1).mean()


def _eval(model, loader, device):
    model.eval()
    sums = defaultdict(float)
    n_total = 0
    per_k = defaultdict(lambda: {"emb": 0.0, "ref": 0.0, "velo": 0.0, "n": 0})
    has_velo_target = False
    with torch.no_grad():
        for batch in loader:
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            pred_emb, pred_ref, pred_velo = model(
                stale_new_emb=batch_d["stale_new_emb"],
                stale_new_ref=batch_d["stale_new_ref"],
                stale_new_velo=batch_d["stale_new_velo"],
                carried_emb=batch_d["carried_emb"],
                carried_ref=batch_d["carried_ref"],
                k=batch_d["k"],
            )
            emb_l2 = per_token_l2(pred_emb, batch_d["fresh_new_emb"]).item()
            ref_l2 = per_token_l2(pred_ref, batch_d["fresh_new_ref"]).item()
            n = batch_d["fresh_new_emb"].shape[0]
            sums["emb"] += emb_l2 * n
            sums["ref"] += ref_l2 * n
            if "fresh_new_velo" in batch_d:
                has_velo_target = True
                velo_l2 = per_token_l2(pred_velo, batch_d["fresh_new_velo"]).item()
                sums["velo"] += velo_l2 * n
            n_total += n
            # Per-k breakdown
            ks = batch_d["k"].cpu().tolist()
            for i, kv in enumerate(ks):
                bag = per_k[kv]
                bag["emb"] += (pred_emb[i] - batch_d["fresh_new_emb"][i]).norm(dim=-1).mean().item()
                bag["ref"] += (pred_ref[i] - batch_d["fresh_new_ref"][i]).norm(dim=-1).mean().item()
                if "fresh_new_velo" in batch_d:
                    bag["velo"] += (pred_velo[i] - batch_d["fresh_new_velo"][i]).norm(dim=-1).mean().item()
                bag["n"]   += 1
    mean_emb = sums["emb"] / max(n_total, 1)
    mean_ref = sums["ref"] / max(n_total, 1)
    mean_velo = (sums["velo"] / max(n_total, 1)) if has_velo_target else None
    per_k_summary = {}
    for kv, v in per_k.items():
        entry = {"emb": v["emb"] / v["n"], "ref": v["ref"] / v["n"], "n": v["n"]}
        if has_velo_target:
            entry["velo"] = v["velo"] / v["n"]
        per_k_summary[kv] = entry
    return mean_emb, mean_ref, mean_velo, per_k_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="output of dump_ltc_pairs.py")
    ap.add_argument("--ckpt-out", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-ref", type=float, default=1.0)
    ap.add_argument("--lambda-velo", type=float, default=1.0,
                    help="weight on MSE(pred_velo, fresh_new_velo). "
                         "Used only when --predict-velo is set.")
    ap.add_argument("--predict-velo", action="store_true",
                    help="v3 mode: also predict memory_velo so all three "
                         "head-state fields (emb, ref, velo) move "
                         "together. Requires fresh_new_velo in pairs.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    blob = torch.load(args.pairs, map_location="cpu", weights_only=False)
    pairs = blob["pairs"]
    print(f"Loaded {len(pairs)} pairs (k ∈ {sorted(set(p['k'] for p in pairs))})")

    train_pairs, val_pairs, val_scenes = scene_split(pairs, args.val_frac, args.seed)
    print(f"Train: {len(train_pairs)} pairs / Val: {len(val_pairs)} pairs "
          f"({len(val_scenes)} held-out scenes)")

    if args.predict_velo and "fresh_new_velo" not in pairs[0]:
        raise SystemExit(
            "--predict-velo requires pairs dumped with fresh_new_velo. "
            "Re-dump with the v3 dump_ltc_pairs.py."
        )

    train_ds = LTCPairsDataset(train_pairs)
    val_ds   = LTCPairsDataset(val_pairs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate, drop_last=False)
    eval_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collate, drop_last=False)
    eval_val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate, drop_last=False)

    model = LTC(LTCConfig(dropout=args.dropout,
                          predict_velo=args.predict_velo)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LTC: {n_params:,} params  (predict_velo={args.predict_velo})")

    # Init baseline (vel_warp): emb_l2 and ref_l2 at zero-delta.
    base_emb_t, base_ref_t, base_velo_t, _ = _eval(model, eval_train_loader, device)
    base_emb_v, base_ref_v, base_velo_v, _ = _eval(model, eval_val_loader, device)
    base_str_t = (f"emb={base_emb_t:.4f} ref={base_ref_t:.4f}"
                  + (f" velo={base_velo_t:.4f}" if base_velo_t is not None else ""))
    base_str_v = (f"emb={base_emb_v:.4f} ref={base_ref_v:.4f}"
                  + (f" velo={base_velo_v:.4f}" if base_velo_v is not None else ""))
    print(f"Init (= vel_warp baseline) | train: {base_str_t}  | val: {base_str_v}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            pred_emb, pred_ref, pred_velo = model(
                stale_new_emb=batch_d["stale_new_emb"],
                stale_new_ref=batch_d["stale_new_ref"],
                stale_new_velo=batch_d["stale_new_velo"],
                carried_emb=batch_d["carried_emb"],
                carried_ref=batch_d["carried_ref"],
                k=batch_d["k"],
            )
            l_emb = ((pred_emb - batch_d["fresh_new_emb"]) ** 2).sum(-1).mean()
            l_ref = ((pred_ref - batch_d["fresh_new_ref"]) ** 2).sum(-1).mean()
            loss = l_emb + args.lambda_ref * l_ref
            if args.predict_velo:
                l_velo = ((pred_velo - batch_d["fresh_new_velo"]) ** 2).sum(-1).mean()
                loss = loss + args.lambda_velo * l_velo

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                e_t, r_t, v_t, _ = _eval(model, eval_train_loader, device)
                e_v, r_v, v_v, _ = _eval(model, eval_val_loader, device)
                model.train()
                dt = time.time() - t0
                velo_t_str = f" velo={v_t:.4f}" if v_t is not None else ""
                velo_v_str = f" velo={v_v:.4f}" if v_v is not None else ""
                print(f"step {step:5d}  train emb={e_t:.4f} ref={r_t:.4f}{velo_t_str}  "
                      f"| val emb={e_v:.4f} ref={r_v:.4f}{velo_v_str}  "
                      f"(loss={loss.item():.3f}, {dt:.1f}s)")

    # Final eval — per-k breakdown
    e_t, r_t, v_t, by_k_t = _eval(model, eval_train_loader, device)
    e_v, r_v, v_v, by_k_v = _eval(model, eval_val_loader, device)
    print()
    final_t_str = f"emb={e_t:.4f}  ref={r_t:.4f}" + (f"  velo={v_t:.4f}" if v_t is not None else "")
    final_v_str = f"emb={e_v:.4f}  ref={r_v:.4f}" + (f"  velo={v_v:.4f}" if v_v is not None else "")
    base_t_str = f"emb={base_emb_t:.4f} ref={base_ref_t:.4f}" + (f" velo={base_velo_t:.4f}" if base_velo_t is not None else "")
    base_v_str = f"emb={base_emb_v:.4f} ref={base_ref_v:.4f}" + (f" velo={base_velo_v:.4f}" if base_velo_v is not None else "")
    print(f"FINAL train: {final_t_str}  (baseline {base_t_str})")
    print(f"FINAL  val : {final_v_str}  (baseline {base_v_str})")
    print()
    print("Per-k val breakdown:")
    for kv in sorted(by_k_v):
        s = by_k_v[kv]
        extra = f"  velo={s['velo']:.4f}" if "velo" in s else ""
        print(f"  k={kv}: emb={s['emb']:.4f}  ref={s['ref']:.4f}{extra}  (n={s['n']})")

    os.makedirs(os.path.dirname(args.ckpt_out) or ".", exist_ok=True)
    torch.save({
        "state_dict":   model.state_dict(),
        "cfg":          model.cfg.__dict__,
        "val_scenes":   val_scenes,
        "baseline":     {"train": (base_emb_t, base_ref_t, base_velo_t),
                         "val":   (base_emb_v, base_ref_v, base_velo_v)},
        "final":        {"train": (e_t, r_t, v_t),
                         "val":   (e_v, r_v, v_v),
                         "per_k_val": by_k_v},
    }, args.ckpt_out)
    print(f"\nSaved → {args.ckpt_out}")


if __name__ == "__main__":
    main()
