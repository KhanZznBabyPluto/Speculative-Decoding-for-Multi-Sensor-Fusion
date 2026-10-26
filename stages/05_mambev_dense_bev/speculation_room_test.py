"""Stage 05 — Phase 2.3: is there learnable speculation room in MamBEV's BEV?

Phase 2.2 measured `cos(BEV_t, BEV_{t+1}) = 0.885 ± 0.05` across 845 pairs.
That leaves a ~5-25 % per-cell varying residual. The cache framing
says this residual is unpredictable from BEV(t) alone (i.e., new image
content). The speculation framing says it's predictable. This script
arbitrates.

Test:
  Given BEV(t), predict the residual `BEV(t+1) - BEV(t)`. Train a
  tiny conv on 2 of our 3 dumped pairs, evaluate on the held-out pair.
  Compare per-cell L2 of the learned prediction vs the trivial baseline
  `predict_zero` (== `copy_prev_bev`).

Outcomes:
  * Learned L2 < zero-baseline L2 on held-out pair → there IS learnable
    structure. Speculation hypothesis survives. Next: scale up data,
    test if the L2 win translates through MamBEV's head (Phase 4-style
    chimera risk).
  * Learned L2 ≈ zero-baseline → no learnable signal. Cache framing
    dominant. Stage 05 closes definitively negative.
  * Learned L2 >> zero-baseline → bug.

Train data: 2 dumps × 1 pair each × 2 frames each = 40 K cells × 256 ch
Held-out: 1 dump × 1 pair × 2 frames = 20 K cells.
Tiny by ML standards but enough for a smell-test signal.

Model: 3-layer conv, ~150 K params, zero-init final layer so the model
starts at exactly the `predict_zero` baseline. Training can only reduce
loss from there.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BEV_H, BEV_W, BEV_C = 100, 100, 256

DEFAULT_DUMPS = [
    "/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/bev_pair.pt",
    "/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/bev_pair_scene2.pt",
    "/ibex/user/khangea/multisensor/stages/05_mambev_dense_bev/dumps/bev_pair_scene3.pt",
]


def reshape_bev(bev_flat: torch.Tensor) -> torch.Tensor:
    """(groups, H*W, C) → (groups, C, H, W). Take group 0 only."""
    if bev_flat.dim() == 3:
        g, n, c = bev_flat.shape
        return bev_flat.view(g, BEV_H, BEV_W, c).permute(0, 3, 1, 2).contiguous().float()[0:1]
    raise ValueError(bev_flat.shape)


def load_pair(path: Path) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Return (BEV_t, BEV_{t+1}, ego_motion_magnitude). Both (1, C, H, W)."""
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    bev_t  = reshape_bev(d["bev_t"])
    bev_t1 = reshape_bev(d["bev_t+1"])
    eg_t   = d["ego2global_t"]
    eg_t1  = d["ego2global_t+1"]
    import numpy as np
    T = np.linalg.inv(eg_t) @ eg_t1
    ego_mag = float(np.linalg.norm(T[:3, 3]))
    return bev_t, bev_t1, ego_mag


class TinySpeculator(nn.Module):
    """Tiny conv: 3 layers of 1×1 + 3×3 + 1×1 convs. Zero-init final layer
    so the initial prediction is exactly zero residual (matches the
    copy_prev_bev baseline). Loss can only go down from there."""
    def __init__(self, C=BEV_C, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(C, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(hidden, C, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, bev_t):
        return bev_t + self.head(self.body(bev_t))


def per_cell_l2(pred, target):
    """(B, C, H, W) → (B, H, W) per-cell L2 norm of channel-difference."""
    return torch.sqrt(((pred - target) ** 2).sum(dim=1) + 1e-12)


def cosine_per_cell(pred, target):
    eps = 1e-6
    num = (pred * target).sum(dim=1)
    dp  = torch.sqrt((pred * pred).sum(dim=1) + eps)
    dt  = torch.sqrt((target * target).sum(dim=1) + eps)
    return num / (dp * dt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", type=Path, nargs="+",
                    default=[Path(p) for p in DEFAULT_DUMPS])
    ap.add_argument("--held-out-idx", type=int, default=2,
                    help="which dump index to use as held-out (0/1/2)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr",    type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load all pairs ----
    pairs = []
    for p in args.dumps:
        b_t, b_t1, ego = load_pair(p)
        pairs.append((b_t, b_t1, ego, p.name))
        print(f"  loaded {p.name}: ego={ego:.3f}m, "
              f"BEV(t).shape={tuple(b_t.shape)}")

    train = [p for i, p in enumerate(pairs) if i != args.held_out_idx]
    val   = pairs[args.held_out_idx]
    print(f"\nTrain: {[p[3] for p in train]}")
    print(f"Val  : {val[3]}\n")

    # Stack train pairs along batch dim
    train_t  = torch.cat([p[0] for p in train], dim=0).to(device)
    train_t1 = torch.cat([p[1] for p in train], dim=0).to(device)
    val_t   = val[0].to(device)
    val_t1  = val[1].to(device)

    # ---- Baselines (predict_zero == copy_prev_bev) ----
    with torch.no_grad():
        base_train_l2 = per_cell_l2(train_t, train_t1).mean().item()
        base_val_l2   = per_cell_l2(val_t,   val_t1).mean().item()
        base_train_cos = cosine_per_cell(train_t, train_t1).mean().item()
        base_val_cos   = cosine_per_cell(val_t,   val_t1).mean().item()
    print(f"BASELINE (predict_zero = copy_prev_bev):")
    print(f"  train L2 = {base_train_l2:.4f}   cos = {base_train_cos:.4f}")
    print(f"  val   L2 = {base_val_l2:.4f}   cos = {base_val_cos:.4f}")

    # ---- Train ----
    model = TinySpeculator(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"\nSpeculator: {n_params/1e3:.1f}K params  steps={args.steps}  lr={args.lr}\n")

    t_start = time.time()
    for step in range(1, args.steps + 1):
        pred = model(train_t)
        loss = ((pred - train_t1) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step in (1, 50) or step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                tr_l2 = per_cell_l2(model(train_t), train_t1).mean().item()
                va_l2 = per_cell_l2(model(val_t),   val_t1).mean().item()
                va_cos = cosine_per_cell(model(val_t), val_t1).mean().item()
            print(f"step {step:5d}  loss={loss.item():.4f}  "
                  f"train L2={tr_l2:.4f}  val L2={va_l2:.4f}  "
                  f"val cos={va_cos:.4f}")
            model.train()

    elapsed = time.time() - t_start
    print(f"\nTraining: {elapsed:.1f}s")

    # ---- Final ----
    model.eval()
    with torch.no_grad():
        final_tr_l2 = per_cell_l2(model(train_t), train_t1).mean().item()
        final_va_l2 = per_cell_l2(model(val_t),   val_t1).mean().item()
        final_va_cos = cosine_per_cell(model(val_t), val_t1).mean().item()

    rel_train = 100.0 * (base_train_l2 - final_tr_l2) / max(base_train_l2, 1e-9)
    rel_val   = 100.0 * (base_val_l2   - final_va_l2) / max(base_val_l2,   1e-9)

    print("\n" + "=" * 60)
    print(f"BASELINE          train L2={base_train_l2:.4f}   val L2={base_val_l2:.4f}")
    print(f"LEARNED           train L2={final_tr_l2:.4f}   val L2={final_va_l2:.4f}")
    print(f"  improvement:    train {rel_train:+.2f}%      val {rel_val:+.2f}%")
    print(f"  val cosine: {base_val_cos:.4f} → {final_va_cos:.4f}")
    print("=" * 60)

    if rel_val > 5.0:
        print("\n→ Speculation room EXISTS. Held-out L2 dropped meaningfully.")
        print("  Phase 4 caveat: L2 reduction may not transfer to mAP through")
        print("  MamBEV's head (chimera risk). Needs a head-passthrough test next.")
    elif rel_val > 0.0:
        print("\n→ Marginal learnable signal. Likely scene-specific overfitting")
        print("  on 2 pairs; needs more data to be conclusive.")
    else:
        print("\n→ NO learnable signal beyond copy_prev_bev. The BEV residual")
        print("  is dominated by content unpredictable from BEV(t) alone")
        print("  (likely new image features). Cache framing wins; stage 05 closes")
        print("  with a clean structural negative.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
