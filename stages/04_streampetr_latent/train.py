"""Phase 3.3: speculator overfit smoke test.

Trains the Phase 3.2 speculator on a single scene (39 frame pairs at
k=1, all-fresh schedule). If the architecture is sound, the model
should overfit easily — driving the new-block embedding L2 well below
the Phase 2 trivial baseline of 4.24 and refpoint L2 well below 36 m.

If it cannot overfit a 39-pair training set, the architecture has a
bug (PE not helping, decoder too small, normalization broken, etc.) —
fix it BEFORE scale-up.

## Loss

Mean per-token L2 over the new block, matching the Phase 2 metric:

    L_emb = mean_token  ||pred_emb - tgt_new_emb||_2          (unit-less)
    L_ref = mean_token  ||pred_ref - tgt_new_ref||_2          (metres)
    L     = L_emb + lambda_ref · L_ref

`lambda_ref = 0.1` balances the two roughly at init (L_emb≈4.2 vs
L_ref≈36 → contributions ≈ 4.2 + 3.6 = 7.8). The metric we report is
each component separately so we can compare to Phase 2 head-on.

## Run

    cd stages/04_streampetr_latent
    python train.py --dump dumps/scene_0.pt --steps 600 --batch-size 8
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataset import (DatasetConfig, SpeculatorDataset, collate,           # noqa: E402
                     REF_SCALE, make_train_val_split)
from speculator import Speculator, SpeculatorConfig                       # noqa: E402


def per_token_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over tokens of L2 norm over the last dim. Matches Phase 2 metric."""
    return (pred - target).norm(dim=-1).mean()


def mse_sum_over_dim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared L2 per token, averaged over tokens. Stronger gradient signal
    far from target than the unit-magnitude gradient of mean-L2."""
    return ((pred - target) ** 2).sum(dim=-1).mean()


def _zero_carried(batch):
    """Diagnostic: zero out carried_emb / carried_ref so the model is forced
    to predict from img_feats alone. See --zero-carried CLI flag."""
    batch["carried_emb"] = torch.zeros_like(batch["carried_emb"])
    batch["carried_ref"] = torch.zeros_like(batch["carried_ref"])
    return batch


def _eval_full(model, loader, device, zero_carried=False):
    """Evaluate Phase 2-comparable metrics: mean per-token L2 for both
    embedding and refpoint. Refpoint is reported in metres (unscaled)."""
    model.eval()
    sums = {"emb": 0.0, "ref_m": 0.0, "n": 0}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if zero_carried:
                batch = _zero_carried(batch)
            pred_emb, pred_ref = model(
                carried_emb=batch["carried_emb"],
                carried_ref=batch["carried_ref"],
                img_feats=batch["img_feats"],
                staleness=batch["staleness"],
                src_new_emb=batch["src_new_emb"],
                src_new_ref=batch["src_new_ref"],
            )
            emb_l2 = per_token_l2(pred_emb, batch["tgt_new_emb"]).item()
            # Refs were divided by REF_SCALE in the dataset → multiply back.
            ref_l2_m = per_token_l2(pred_ref, batch["tgt_new_ref"]).item() * REF_SCALE
            n = batch["tgt_new_emb"].shape[0]
            sums["emb"] += emb_l2 * n
            sums["ref_m"] += ref_l2_m * n
            sums["n"] += n
    return sums["emb"] / sums["n"], sums["ref_m"] / sums["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True,
                    help="one or more dump .pt files")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-ref", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="0 = no clipping (default for overfit smoke)")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=1,
                    help="how many frames ahead to predict (uniform over [k_min, k_max])")
    ap.add_argument("--schedule-mode", choices=("fresh", "random"), default="fresh",
                    help="'fresh' = all cams arrive on time; 'random' = sample per-cam staleness")
    ap.add_argument("--max-staleness", type=int, default=1,
                    help="upper bound on per-cam staleness when schedule-mode=random")
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="speculator decoder dropout (0.0 to disable for overfit/debug)")
    ap.add_argument("--val-scenes", type=int, default=0,
                    help="hold out the last N scenes for validation "
                         "(0 = no split, train on everything)")
    ap.add_argument("--zero-carried", action="store_true",
                    help="zero out carried_emb and carried_ref before they reach "
                         "the speculator. Diagnostic: forces the model to use "
                         "ONLY img_feats. If val emb drops below baseline with "
                         "this flag, carried_emb path was the generalization "
                         "contaminator. If val still grows, task is harder.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ---------------------------------------------------------------
    dump_paths = tuple(args.dump)
    if args.val_scenes > 0:
        train_ds, val_ds = make_train_val_split(
            dump_paths=dump_paths, n_val_scenes=args.val_scenes,
            k_min=args.k_min, k_max=args.k_max,
            schedule_mode=args.schedule_mode,
            max_staleness=args.max_staleness, seed=args.seed,
        )
        print(f"Train: {len(train_ds)} pairs from "
              f"{len(train_ds.active_scene_tokens)} scene(s); "
              f"Val: {len(val_ds)} pairs from "
              f"{len(val_ds.active_scene_tokens)} scene(s)")
    else:
        train_ds = SpeculatorDataset(DatasetConfig(
            dump_paths=dump_paths,
            k_min=args.k_min, k_max=args.k_max,
            schedule_mode=args.schedule_mode,
            max_staleness=args.max_staleness, seed=args.seed,
        ))
        val_ds = None
        print(f"Train: {len(train_ds)} pairs from "
              f"{len(train_ds.active_scene_tokens)} scene(s); "
              f"Val: (none — no scene split)")

    print(f"  (k in [{args.k_min},{args.k_max}], schedule={args.schedule_mode}, "
          f"max_staleness={args.max_staleness if args.schedule_mode == 'random' else 0})")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=False)
    eval_train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0,
                                   collate_fn=collate, drop_last=False)
    eval_val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=0, collate_fn=collate, drop_last=False)
        if val_ds is not None else None
    )

    # ---- Model --------------------------------------------------------------
    model = Speculator(SpeculatorConfig(dropout=args.dropout)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params")

    # ---- Baseline (copy_prev_new) at init ----------------------------------
    base_emb_t, base_ref_t = _eval_full(model, eval_train_loader, device,
                                        zero_carried=args.zero_carried)
    if eval_val_loader is not None:
        base_emb_v, base_ref_v = _eval_full(model, eval_val_loader, device,
                                            zero_carried=args.zero_carried)
        print(f"Init baseline  | train: emb={base_emb_t:.4f}  ref={base_ref_t:.2f} m"
              f"  | val: emb={base_emb_v:.4f}  ref={base_ref_v:.2f} m")
    else:
        base_emb_v, base_ref_v = base_emb_t, base_ref_t
        print(f"Init baseline  | train: emb={base_emb_t:.4f}  ref={base_ref_t:.2f} m")

    # ---- Train --------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if args.zero_carried:
                batch = _zero_carried(batch)
            pred_emb, pred_ref = model(
                carried_emb=batch["carried_emb"],
                carried_ref=batch["carried_ref"],
                img_feats=batch["img_feats"],
                staleness=batch["staleness"],
                src_new_emb=batch["src_new_emb"],
                src_new_ref=batch["src_new_ref"],
            )
            l_emb_mse = mse_sum_over_dim(pred_emb, batch["tgt_new_emb"])
            l_ref_mse = mse_sum_over_dim(pred_ref, batch["tgt_new_ref"])
            loss = l_emb_mse + args.lambda_ref * l_ref_mse

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                e_t, r_t = _eval_full(model, eval_train_loader, device,
                                      zero_carried=args.zero_carried)
                if eval_val_loader is not None:
                    e_v, r_v = _eval_full(model, eval_val_loader, device,
                                          zero_carried=args.zero_carried)
                    line = (f"step {step:5d}  train emb={e_t:.4f} ref={r_t:.2f}m  "
                            f"| val emb={e_v:.4f} ref={r_v:.2f}m")
                else:
                    line = f"step {step:5d}  train emb={e_t:.4f} ref={r_t:.2f}m"
                model.train()
                dt = time.time() - t0
                print(f"{line}  (loss={loss.item():.4f}, {dt:.1f}s)")

    # ---- Final eval ---------------------------------------------------------
    e_t, r_t = _eval_full(model, eval_train_loader, device,
                          zero_carried=args.zero_carried)
    print()
    print(f"Final TRAIN: emb_l2={e_t:.4f}  ref_l2={r_t:.4f} m"
          f"  (baseline emb={base_emb_t:.4f} ref={base_ref_t:.2f} m)")
    if eval_val_loader is not None:
        e_v, r_v = _eval_full(model, eval_val_loader, device,
                              zero_carried=args.zero_carried)
        print(f"Final VAL  : emb_l2={e_v:.4f}  ref_l2={r_v:.4f} m"
              f"  (baseline emb={base_emb_v:.4f} ref={base_ref_v:.2f} m)")
        print(f"\nPhase 2 floor (copy_prev_new, k=1): emb_l2≈4.24  ref_l2≈36.15 m")
        gen_gap_emb = e_v - e_t
        gen_gap_ref = r_v - r_t
        print(f"Generalization gap: Δemb={gen_gap_emb:+.4f}  Δref={gen_gap_ref:+.2f} m")
        if e_v < base_emb_v * 0.9:
            print("OK — val gap below baseline (speculator generalizes)")
        else:
            print("WARNING — val gap ≥ 90% of baseline (model memorizes scenes, "
                  "not function)")
    else:
        print(f"Phase 2 floor (copy_prev_new, k=1): emb_l2≈4.24  ref_l2≈36.15 m")
        if e_t < base_emb_t * 0.5:
            print("OK — train metrics reduced by ≥ 50% (architecture sound)")
        else:
            print("WARNING — train did not reduce metrics enough; investigate.")


if __name__ == "__main__":
    main()
