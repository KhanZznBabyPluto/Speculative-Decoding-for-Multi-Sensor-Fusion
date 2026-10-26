"""Phase 4 — Learned Temporal Calibration (LTC) speculator.

Given the head's stale-derived new memory block (embedding + ref +
velocity), the staleness count k, and the carried memory block as
context, LTC outputs a corrected new block. The correction is applied
on top of the analytical `vel_warp` baseline (`new_ref += velo · dt · k`)
so that at initialization the model exactly reproduces the proven
analytical floor.

Architecture (small by design — ~2-3M params):

  stale_new = (emb, ref, velo) → token  : per-token projection
  + staleness embedding                  : scalar k → vec
  + cross-attention to carried block     : scene context
  → 2-3 decoder layers
  → (Δemb, Δref) heads                   : zero-init, residual on baseline

At init: Δemb = 0, Δref = 0 → output = (stale_emb, stale_ref + velo·dt·k)
which is exactly `vel_warp`. So loss can only go down from the
analytical baseline as training progresses.

The design choice to predict a residual OVER `vel_warp` (not over raw
stale) is informed by Phase 3.4: analytical velocity projection
captures the dominant correction (~3-5% mAP gain); learned residual
fills the remaining gap (object motion non-linearities, head dynamics).

Why this should generalize cross-scene where Phase 3.3 failed:
* Phase 3.3 target: absolute `tgt_new_emb`. Highly scene-conditional
  (the head's queries attend to scene-specific image content).
* Phase 4 target: `fresh_new - vel_warp(stale_new)` — a small
  *correction* over a near-correct analytical baseline. Scene-
  invariant components (object kinematics, head's perturbation
  response) dominate; scene identity becomes secondary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


NUSC_DT_SECONDS = 0.5  # nuScenes keyframe rate


@dataclass
class LTCConfig:
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_staleness: int = 8
    num_new: int = 256
    num_carried: int = 1024
    # v3 (MLN-consistent): also predict memory_velo so all three head-state
    # fields (emb, ref, velo) move together. v2 checkpoints have this False
    # and only the (emb, ref) heads are saved/loaded.
    predict_velo: bool = False


class LTC(nn.Module):
    def __init__(self, cfg: LTCConfig = None):
        super().__init__()
        cfg = cfg or LTCConfig()
        self.cfg = cfg

        # Per-token input projection: stale state (emb + ref + velo) → token
        self.input_proj = nn.Linear(cfg.d_model + 3 + 2, cfg.d_model)
        self.stale_emb = nn.Embedding(cfg.max_staleness + 1, cfg.d_model)

        # Carried-memory PE: ref point MLP (3 → D)
        self.ref_pe_mlp = nn.Sequential(
            nn.Linear(3, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model),
        )
        self.mem_norm = nn.LayerNorm(cfg.d_model)

        # Decoder (cross-attention from stale tokens to carried context)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.n_layers)

        # Residual heads — zero-init so output == vel_warp baseline at step 0
        self.head_emb_delta = nn.Linear(cfg.d_model, cfg.d_model)
        self.head_ref_delta = nn.Linear(cfg.d_model, 3)
        heads = [self.head_emb_delta, self.head_ref_delta]
        if cfg.predict_velo:
            self.head_velo_delta = nn.Linear(cfg.d_model, 2)
            heads.append(self.head_velo_delta)
        for m in heads:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        stale_new_emb: torch.Tensor,   # [B, num_new, D]
        stale_new_ref: torch.Tensor,   # [B, num_new, 3]  centered ego-frame (or any)
        stale_new_velo: torch.Tensor,  # [B, num_new, 2]
        carried_emb: torch.Tensor,     # [B, num_carried, D]
        carried_ref: torch.Tensor,     # [B, num_carried, 3]  same frame as stale_new_ref
        k: torch.Tensor,               # [B] int — frames stale
    ):
        cfg = self.cfg
        B = stale_new_emb.shape[0]

        # --- per-token input encoding ---
        tok_input = torch.cat([
            stale_new_emb.float(),
            stale_new_ref.float(),
            stale_new_velo.float(),
        ], dim=-1)                                                       # [B, N, D+5]
        tok = self.input_proj(tok_input)                                 # [B, N, D]
        tok = tok + self.stale_emb(k).unsqueeze(1)                       # broadcast over tokens

        # --- carried context ---
        mem = self.mem_norm(carried_emb.float())
        mem = mem + self.ref_pe_mlp(carried_ref.float())

        out = self.decoder(tgt=tok, memory=mem)                          # [B, N, D]

        delta_emb = self.head_emb_delta(out)                             # [B, N, D]
        delta_ref = self.head_ref_delta(out)                             # [B, N, 3]

        # Analytical vel_warp baseline: only xy gets the velocity push.
        # KNOWN FRAME BUG (kept so trained checkpoints stay reproducible): at
        # the call site stale_new_ref is GLOBAL-frame but stale_new_velo is
        # lidar(t)-frame, so this push points in the wrong direction. The
        # learned delta was trained on top of it. A correct LTC needs the velo
        # rotated first (measure_stale_map_mitigation.velo_lidar_to_global)
        # AND re-dumped pairs + retraining — not a drop-in fix.
        # k is [B] of ints; broadcast appropriately.
        dt_k = (NUSC_DT_SECONDS * k.float()).view(B, 1, 1)               # [B, 1, 1]
        vw_baseline_ref = stale_new_ref.float().clone()
        vw_baseline_ref[..., :2] = vw_baseline_ref[..., :2] + stale_new_velo.float() * dt_k

        pred_emb = stale_new_emb.float() + delta_emb                     # residual over stale
        pred_ref = vw_baseline_ref + delta_ref                           # residual over vel_warp

        # Velocity prediction (v3, MLN-consistent). When the flag is off,
        # we return the stale velo unchanged so callers can uniformly
        # consume a 3-tuple without branching.
        if cfg.predict_velo:
            delta_velo = self.head_velo_delta(out)                       # [B, N, 2]
            pred_velo = stale_new_velo.float() + delta_velo
        else:
            pred_velo = stale_new_velo.float()
        return pred_emb, pred_ref, pred_velo


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _self_test():
    """CPU shape + zero-init test. Run with `python ltc.py`."""
    torch.manual_seed(0)
    cfg = LTCConfig()
    ltc = LTC(cfg).eval()
    n_params = count_params(ltc)
    print(f"LTC parameters: {n_params:,} (~{n_params / 1e6:.2f}M)")

    B = 4
    stale_emb = torch.randn(B, cfg.num_new, cfg.d_model)
    stale_ref = torch.randn(B, cfg.num_new, 3) * 50
    stale_velo = torch.randn(B, cfg.num_new, 2) * 5
    carried_emb = torch.randn(B, cfg.num_carried, cfg.d_model)
    carried_ref = torch.randn(B, cfg.num_carried, 3) * 50
    k = torch.randint(1, cfg.max_staleness + 1, (B,))

    with torch.no_grad():
        pred_emb, pred_ref, pred_velo = ltc(stale_emb, stale_ref, stale_velo,
                                            carried_emb, carried_ref, k)
    assert pred_emb.shape == (B, cfg.num_new, cfg.d_model), pred_emb.shape
    assert pred_ref.shape == (B, cfg.num_new, 3), pred_ref.shape
    assert pred_velo.shape == (B, cfg.num_new, 2), pred_velo.shape
    print(f"[OK] output shapes: emb={tuple(pred_emb.shape)}, "
          f"ref={tuple(pred_ref.shape)}, velo={tuple(pred_velo.shape)}")

    # Zero-init residual heads → output must equal vel_warp baseline
    dt_k = (NUSC_DT_SECONDS * k.float()).view(B, 1, 1)
    expected_ref = stale_ref.float().clone()
    expected_ref[..., :2] = expected_ref[..., :2] + stale_velo.float() * dt_k
    assert torch.allclose(pred_emb, stale_emb.float(), atol=1e-6)
    assert torch.allclose(pred_ref, expected_ref, atol=1e-5)
    assert torch.allclose(pred_velo, stale_velo.float(), atol=1e-6)
    print("[OK] zero-init delta heads reproduce vel_warp baseline exactly "
          "(and pred_velo == stale_velo when predict_velo=False)")

    # v3 path: predict_velo=True
    cfg_v3 = LTCConfig(predict_velo=True)
    ltc_v3 = LTC(cfg_v3).eval()
    with torch.no_grad():
        _, _, pred_velo_v3 = ltc_v3(stale_emb, stale_ref, stale_velo,
                                    carried_emb, carried_ref, k)
    assert torch.allclose(pred_velo_v3, stale_velo.float(), atol=1e-6), \
        "v3 zero-init head_velo_delta must give pred_velo == stale_velo"
    print(f"[OK] v3 zero-init head_velo_delta: pred_velo == stale_velo "
          f"({count_params(ltc_v3):,} params)")

    # Backward pass test
    ltc.train()
    pred_emb, pred_ref, pred_velo = ltc(stale_emb, stale_ref, stale_velo,
                                        carried_emb, carried_ref, k)
    loss = pred_emb.pow(2).mean() + pred_ref.pow(2).mean()
    loss.backward()
    n_grads = sum(1 for p in ltc.parameters() if p.grad is not None)
    print(f"[OK] backward: gradients flowed to {n_grads} tensors")

    print("\nAll LTC self-tests pass.")


if __name__ == "__main__":
    _self_test()
