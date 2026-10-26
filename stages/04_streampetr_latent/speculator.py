"""Phase 3.2: speculator architecture spec.

Latency-correction module: predicts the NEW-block of StreamPETR's memory
state at frame t+k given (a) the carried memory from frame t and (b)
the camera features available at t+k (some of which may be stale per
the Phase 2 AsyncSchedule). The speculator does NOT model the carried
block — Phase 2 proved that's mechanically perfect — only the new
[0:256] tokens.

## Design (locked in 2026-05-14 discussion)

* Output is a **delta over copy_prev_new** (= src[0:256]). At init the
  delta heads are zero-initialized, so `pred == src_new` (= the Phase 2
  trivial baseline at 4.24 mean L2). The speculator can only improve
  from there; loss can only go down. This is a deliberate residual-
  prediction setup, not a from-scratch regressor.

* **Input normalization is non-optional.** Phase 3.1 measured raw FPN
  output at std≈46, range≈±400. Without LayerNorm at the input the
  first decoder layer will saturate. Two norms: one over the image
  feature channel dim, one over the carried-memory channel dim.

* **Camera spatial PE is learnable, per-cam.** Without PE the
  cross-attention cannot distinguish CAM_FRONT from CAM_BACK or
  near-pixel from far-pixel within a cam. Single learnable
  `[N_cams, H*W, D]` parameter.

* **Carried-memory PE is the reference point** itself, projected
  through a small MLP. The carried embedding has no intrinsic
  positional content; the head normally injects 3D position via
  cross-attention's PE. Speculator mirrors that.

* **Staleness embedding** is a learned `Embedding(max_staleness+1, D)`
  added per-cam after spatial PE. So the cross-attention can implicitly
  down-weight stale cams without us hard-coding a mask.

* **Architecture size**: 2 decoder layers, D=256, 8 heads, FFN×4.
  Roughly 2M parameters. The point is NOT to compete with the head
  for capacity — it's to learn a *correction* on top of a near-correct
  trivial baseline.

## Not in this module

* AsyncSchedule integration — that goes in the training data pipeline
  (Phase 3.3), where for each batch we sample a schedule and substitute
  stale camera features from earlier dumped frames.

* Ego delta — carried memory ref points are in GLOBAL frame and don't
  drift between frames (Phase 1.2 (B)). Ego motion is implicit in the
  refpoint values themselves; no separate ego-delta input needed.

* Reference-point coordinate normalization — global coords are ~1000s
  of metres. The MLP that turns refpoints into PE must see *centered*
  coords to train stably. Phase 3.3 will subtract `ego_pose_translation`
  before feeding refs into the speculator. For the unit test in this
  module we generate synthetic ~unit-scale refs, so the issue doesn't
  appear yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SpeculatorConfig:
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    n_cams: int = 6
    h_feat: int = 16
    w_feat: int = 44
    max_staleness: int = 8
    num_new: int = 256          # head's num_propagated
    num_carried: int = 1024     # head's memory_len


class Speculator(nn.Module):
    def __init__(self, cfg: SpeculatorConfig = None):
        super().__init__()
        cfg = cfg or SpeculatorConfig()
        self.cfg = cfg

        # Input normalization
        self.feat_norm = nn.LayerNorm(cfg.d_model)
        self.mem_norm = nn.LayerNorm(cfg.d_model)

        # Image feature projection + learnable spatial PE per cam
        self.cam_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.img_pe = nn.Parameter(
            torch.randn(cfg.n_cams, cfg.h_feat * cfg.w_feat, cfg.d_model) * 0.02
        )
        self.stale_emb = nn.Embedding(cfg.max_staleness + 1, cfg.d_model)

        # Carried-memory PE: ref point MLP (3 → D)
        self.ref_pe_mlp = nn.Sequential(
            nn.Linear(3, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model),
        )

        # Learnable query tokens for the new block
        self.queries = nn.Parameter(torch.randn(cfg.num_new, cfg.d_model) * 0.02)

        # Decoder
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.n_layers)

        # Output delta heads — zero-initialized so model starts at the
        # copy_prev_new baseline.
        self.head_emb_delta = nn.Linear(cfg.d_model, cfg.d_model)
        self.head_ref_delta = nn.Linear(cfg.d_model, 3)
        for m in (self.head_emb_delta, self.head_ref_delta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        carried_emb: torch.Tensor,   # [B, num_carried, D]
        carried_ref: torch.Tensor,   # [B, num_carried, 3]   global, ego-centered
        img_feats: torch.Tensor,     # [B, n_cams, D, H, W]   any dtype
        staleness: torch.Tensor,     # [B, n_cams]            int, in [0, max_staleness]
        src_new_emb: torch.Tensor,   # [B, num_new, D]        copy_prev_new baseline
        src_new_ref: torch.Tensor,   # [B, num_new, 3]
    ):
        cfg = self.cfg
        B = img_feats.shape[0]

        # --- image tokens -------------------------------------------------
        img = img_feats.float()                                     # [B,N,D,H,W]
        img = img.permute(0, 1, 3, 4, 2).contiguous()               # [B,N,H,W,D]
        img = img.reshape(B, cfg.n_cams, cfg.h_feat * cfg.w_feat, cfg.d_model)
        img = self.feat_norm(img)
        img = self.cam_proj(img)
        img = img + self.img_pe.unsqueeze(0)                        # per-cam spatial PE
        img = img + self.stale_emb(staleness).unsqueeze(2)          # per-cam staleness
        img = img.reshape(B, cfg.n_cams * cfg.h_feat * cfg.w_feat, cfg.d_model)

        # --- carried memory tokens ---------------------------------------
        mem = self.mem_norm(carried_emb.float())
        mem = mem + self.ref_pe_mlp(carried_ref.float())

        # --- KV set + queries --------------------------------------------
        kv = torch.cat([img, mem], dim=1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()

        # --- decode -------------------------------------------------------
        out = self.decoder(tgt=q, memory=kv)

        # --- predict deltas + apply over baseline -------------------------
        pred_emb = src_new_emb.float() + self.head_emb_delta(out)
        pred_ref = src_new_ref.float() + self.head_ref_delta(out)
        return pred_emb, pred_ref


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CPU shape / behavior unit test (`python speculator.py`)
# ---------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    cfg = SpeculatorConfig()
    spec = Speculator(cfg).eval()

    n_params = count_params(spec)
    print(f"Speculator parameters: {n_params:,} (~{n_params/1e6:.2f}M)")

    B = 2
    carried_emb = torch.randn(B, cfg.num_carried, cfg.d_model)
    carried_ref = torch.randn(B, cfg.num_carried, 3) * 50.0     # rough BEV scale
    img_feats = torch.randn(B, cfg.n_cams, cfg.d_model, cfg.h_feat, cfg.w_feat,
                            dtype=torch.float16)
    staleness = torch.randint(0, cfg.max_staleness + 1, (B, cfg.n_cams))
    src_new_emb = torch.randn(B, cfg.num_new, cfg.d_model)
    src_new_ref = torch.randn(B, cfg.num_new, 3) * 50.0

    with torch.no_grad():
        pred_emb, pred_ref = spec(carried_emb, carried_ref, img_feats,
                                  staleness, src_new_emb, src_new_ref)

    assert pred_emb.shape == (B, cfg.num_new, cfg.d_model), pred_emb.shape
    assert pred_ref.shape == (B, cfg.num_new, 3), pred_ref.shape
    print(f"[OK] output shapes: pred_emb={tuple(pred_emb.shape)}  "
          f"pred_ref={tuple(pred_ref.shape)}")

    # Zero-init delta heads → output must equal copy_prev_new baseline.
    assert torch.allclose(pred_emb, src_new_emb.float(), atol=1e-6), \
        "pred_emb != src_new_emb at zero-delta init"
    assert torch.allclose(pred_ref, src_new_ref.float(), atol=1e-6), \
        "pred_ref != src_new_ref at zero-delta init"
    print("[OK] zero-delta init reproduces copy_prev_new baseline exactly")

    # After a parameter perturbation of the delta heads, output should
    # diverge from the baseline.
    with torch.no_grad():
        spec.head_emb_delta.weight.normal_(0, 0.01)
        spec.head_emb_delta.bias.normal_(0, 0.01)
        pred_emb2, _ = spec(carried_emb, carried_ref, img_feats,
                            staleness, src_new_emb, src_new_ref)
    diff = (pred_emb2 - src_new_emb).abs().mean().item()
    assert diff > 1e-6, "delta head perturbation produced no change"
    print(f"[OK] after delta-head perturbation, mean |pred - baseline| = {diff:.4f}")

    # Backward pass works.
    spec.train()
    pred_emb, pred_ref = spec(carried_emb, carried_ref, img_feats,
                              staleness, src_new_emb, src_new_ref)
    loss = (pred_emb.pow(2).mean() + pred_ref.pow(2).mean())
    loss.backward()
    n_grads = sum(1 for p in spec.parameters() if p.grad is not None)
    print(f"[OK] backward pass: {n_grads} tensors received gradients")

    print("\nAll speculator self-tests pass.")


if __name__ == "__main__":
    _self_test()
