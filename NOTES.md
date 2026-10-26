# READ ME FIRST — project state and how to read this notebook

This is an append-only lab notebook: older sections are kept as written, and
later sections correct them. Read in this order:

1. **Audit: re-checking every claim from stage 04 onward** — which earlier
   conclusions changed and why (vel_warp, the "1/6 vs 5/6" split, the age port,
   stage-08 rare-class radii). Includes the vel_warp frame-bug re-run.
2. **Stage 09** (realistic 80–500 ms delays, per-camera asynchrony) and
   **Stage 10 → 10e** (delay fine-tuning, age conditioning, lr sweep, mode
   switching), ending in **Stages 09–10: closing summary**.
3. **Stage 08 after the audit fixes** — the corrected envelope numbers. Earlier
   stage-08 sections are superseded where they disagree.

Status by stage:

| stage | topic | status |
|-------|-------|--------|
| 01–03 | box-level speculative drafts (LSTM/MST on BEVFusion/CenterPoint) | closed; code in `archive/stages/` |
| 04 | StreamPETR latent speculation, staleness harness, honest key lift | speculation closed (negative); honest lift stands |
| 05–06 | MamBEV dense-BEV cache statistics, baseline | stands (consistency check only) |
| 07 | trained age port | conclusion weakened by the audit (port started from a harmful init) |
| 08 | conformal risk envelopes, refresh policy | corrected — see "Stage 08 after the audit fixes" |
| 09 | sub-keyframe delay, per-camera async | stands |
| 10 | delay FT (10), age conditioning (10b/c), mode switching (10d/e) | closed; switching is the deployable result |

The opening "Transition Guide: Speculative BEV Dynamics (SSM)" below is the
original, never-executed plan; it is kept for history only. All results use a
23-scene / 914-sample nuScenes val subset (partial blob download), one model
(StreamPETR R50), one seed unless stated.

---

This instructional guide outlines the migration from your current **LSTM-based box prediction** to a **Latent State-Space Model (SSM)** that operates directly on BEV feature maps.

---

# Transition Guide: Speculative BEV Dynamics (SSM)

## 1. Overview of the Shift
*   **Old Logic:** `Previous Boxes -> LSTM -> Predicted Boxes -> Heatmap Sampling -> Accept/Reject`.
*   **New Logic:** `Previous BEV Feature Map -> SSM (Encoder + Conv-Dynamics + Decoder) -> Predicted BEV Feature Map -> Detection Head -> Accept/Reject`.

---

## 2. Updated Model Architecture (`model.py`)
Replace your `TemporalDraftNet` with a Convolutional SSM. This model learns to evolve the "latent world" of the BEV map.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVSSM(nn.Module):
    def __init__(self, bev_channels=256, latent_dim=64):
        super().__init__()
        
        # 1. ENCODER: Compress [256, 180, 180] -> [64, 45, 45]
        self.encoder = nn.Sequential(
            nn.Conv2d(bev_channels, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # 2. DYNAMICS (Option A): Residual Conv Dynamics
        self.dynamics = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        )

        # 3. DECODER: Decompress back to [256, 180, 180]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, bev_channels, 4, stride=2, padding=1)
        )

    def forward(self, bev_t):
        """
        bev_t: The warped BEV tensor from previous frame [1, 256, 180, 180]
        Returns: bev_t1_hat (Predicted next BEV features)
        """
        z_t = self.encoder(bev_t)
        # Learn the residual change (delta) in latent space
        z_t1 = z_t + self.dynamics(z_t)
        bev_t1_hat = self.decoder(z_t1)
        return bev_t1_hat
```

---

## 3. Data Collection Changes (`collect_ssm_data.py`)
**Warning:** Raw BEV tensors are large. Do not save 1 million samples.
*   **Goal:** Save pairs of `(BEV_t_warped, BEV_t1_actual)`.
*   **Process:**
    1. Run `speculator.full_inference` on Frame $T$. Save `bev_embed`.
    2. Move to Frame $T+1$.
    3. Use `get_ego_delta` to warp `bev_embed` to Frame $T+1$'s perspective.
    4. Save the **Warped BEV** as Input ($X$) and the **Actual BEV** (from a full run on Frame $T+1$) as Target ($Y$).

---

## 4. Training Strategy (`train_ssm.py`)
The LSTM used box coordinates. The SSM uses **Feature Reconstruction Loss**.

*   **Loss Function:** 
    1.  `L_feat = MSE(Predicted_BEV, Target_BEV)`
    2.  *(Optional but recommended)* `L_perc = MSE(Head(Predicted_BEV), Head(Target_BEV))` — ensures the predicted features are still useful for the detection head.
*   **Optimizer:** `AdamW` with `lr=1e-4` (SSMs are sensitive to high learning rates).

---

## 5. Inference Pipeline Changes (`smf_final_pipeline.py`)

### Step 1: Feature Propagation
Instead of predicting box coordinates via LSTM:
```python
# current_bev comes from cache (warped)
predicted_bev = ssm_model(current_bev_warped)
```

### Step 2: Verification (The SOTA Way)
In your previous pipeline, you checked specific points. Now you can check the **Consistency** of the whole map:
1.  Apply the `bbox_head` to the `predicted_bev`.
2.  If the `Max Score` from the head on these predicted features is high (e.g., $>0.15$), and the feature-level error is low $\rightarrow$ **ACCEPT**.
3.  If the head is "confused" by the predicted features $\rightarrow$ **REJECT**.

---

## 6. Action Items for the User
1.  **Modify `model.py`:** Add the `BEVSSM` class provided above.
2.  **Create `collect_ssm_data.py`:** Use your existing `SotaSpeculator` to extract and save `bev_embed` tensors from ~50 scenes of `trainval`. 
    *   *Tip:* Use `torch.save(tensor.half())` to save space.
3.  **Run SSM Training:** Train until the reconstructed BEV features look sharp (verify by running the `bbox_head` on them).
4.  **Final Benchmark:** Compare `LSTM + Point Verification` vs `SSM + Latent Verification`.

---

## 7. Key Advantage for your Paper
By moving to **Option A (Conv-SSM)**, you can claim that your system performs **"Spatiotemporal Hallucination"** in latent space. This is much more robust than LSTM because it handles the entire scene context at once, allowing the model to "imagine" where cars will appear or disappear based on surrounding features.

---

# Session 2026-05-13 — Pivot to StreamPETR latent speculation (Stage 04 begins)

> **Status of prior plans.** The earlier transition guide above (BEV-SSM with
> dense `[B, 256, 180, 180]` feature maps from BEVFusion/CenterPoint) and the
> separate `mambev.md` planning doc (now `archive/docs/mambev.md`) (MamBEV backbone) are now *parked*, not
> deleted. The framing has narrowed and matured; what we keep from those docs
> is the underlying intuition (latent-space speculation + feature-consistency
> verification). What we drop is the dense-BEV / MamBEV-specific assumptions.

## Why the reframe

The contribution is **speculative asynchronous perception execution**, not
"better tracker" and not "another world model". Three forces drove the move:

1. **Box-level drafts hit a ceiling.** Stage 03's verifier checks "is the
   object still at the cached BEV's position" while drafts predict "where
   the object is NOW". The mismatch is structural. Scheduled sampling and
   map context (implemented / designed earlier this session) address
   symptoms, not the framing problem.
2. **Reviewers will read box-level work as "yet another detector / tracker".**
   The contribution must be *isolated* — the speculative mechanism on top of
   an *intentionally boring* backbone. CenterPoint is fine as the
   stage-03 backbone for the prior framing; but for the new framing we
   want a base that already *streams* and already maintains *temporal
   latent state*, so the speculative module slots in naturally.
3. **Latent-space verification is cleaner than box-space.** No NMS, no
   Hungarian matching, no thresholded sparse outputs. Latent errors are
   smoother → training is more stable → contribution is more measurable.

## Base model choice: StreamPETR, not BEVFormer / MamBEV

StreamPETR was picked over the alternatives because:

- It already maintains **persistent per-scene memory** across frames
  (`memory_embedding`, `memory_reference_point`, `memory_egopose`,
  `memory_timestamp`, `memory_velo` on `pts_bbox_head`).
- It already assumes **streaming inference**: `prev_scene_token` /
  `reset_memory` are first-class concerns.
- It already operates in a **latent object/state space** — the speculative
  delta fits naturally as a residual on `memory_embedding`.
- The **ego-motion warp is already built in** as a 4×4 point transform
  on `memory_reference_point`. We do not need `grid_sample` and we do
  not need to invent a coordinate convention.
- Public R50-900q weights exist; reproduces 53.7 NDS / 43.2 mAP on
  nuScenes val.

BEVFormer was rejected for this stage as a "useful reference but not the
trunk" — its dense BEV `[B, C, H, W]` matches the previous SSM plan
better, but the streaming abstraction is weaker. MambaBEV / MamBEV was
rejected for now because the Mamba-vs-Transformer choice is orthogonal
to the speculative-execution framing — we can always swap dynamics
backbones later.

## The dense-BEV assumption was wrong

The prior plan (above) assumed `z_t ∈ [B, 256, 180, 180]` from a
BEVFusion-style backbone. **StreamPETR has no such tensor.** Its latent
state is *sparse*:

| Tensor                   | Shape          | Meaning                         |
|--------------------------|----------------|---------------------------------|
| `memory_embedding`       | `[B, M, D]`    | query token features            |
| `memory_reference_point` | `[B, M, 3]`    | 3D position per token (global)  |
| `memory_timestamp`       | `[B, M, 1]`    |                                 |
| `memory_egopose`         | `[B, M, 4, 4]` | ego pose at token capture       |
| `memory_velo`            | `[B, M, 2]`    |                                 |

We confirmed this by reading
[StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py](StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py)
lines 312-374.

**Decision: operate on the sparse memory directly (Option A), not a
rendered dense grid (Option B).** Cleaner, no projection step, matches
the model's native abstraction. The dense `warp_bev` utility from the
original plan is deferred to "only if needed for visualization".

The speculative math from the original plan still holds, just typed
differently:

```
z_{t+1} = z_t + f(z_t, x_camera^{t+1})    # acts on memory_embedding
```

with `memory_reference_point` updated analytically via the SE(3)
warp + a small learned position residual.

## Verifier history (carry-forward — do not relitigate)

From stage 03 (full detail in [archive/stages/03_centerpoint_mst/NOTES.md](archive/stages/03_centerpoint_mst/NOTES.md#verifier-history)):

- The "check `prev_center` instead of `predicted_center`" experiment
  crashed retention to 12-26%. The current `predicted_center` check is
  load-bearing as an implicit reasonableness filter on draft predictions.
- Hybrid output (warped-head + drafts) catastrophically misfires at high
  SPEC fractions.
- These lessons carry into stage 04: **do not decode to boxes during
  speculation**. Verification happens in latent space first (feature
  consistency on `memory_embedding`); the detection head is downstream
  of acceptance, not embedded in it.

## What was done this session for Stage 04

1. **Environment recipe.** StreamPETR needs the legacy OpenMMLab stack
   (mmcv 1.7 / mmdet 2.28 / mmdet3d 1.0rc6 / torch 1.13 + cu117). Created
   shell scripts the user runs on a GPU node — `sensor` env (mmcv 2.x)
   is left untouched so stage 03 stays reproducible.
   - [stages/04_streampetr_latent/env_setup.sh](stages/04_streampetr_latent/env_setup.sh)
   - [stages/04_streampetr_latent/install_mmdet3d.sh](stages/04_streampetr_latent/install_mmdet3d.sh)
   - [stages/04_streampetr_latent/download_checkpoint.sh](stages/04_streampetr_latent/download_checkpoint.sh)
   - [stages/04_streampetr_latent/prepare_data.sh](stages/04_streampetr_latent/prepare_data.sh)

2. **StreamPETR cloned** at [StreamPETR/](StreamPETR/) (depth-1, official
   repo `exiawsh/StreamPETR`). Picked R50-900q config (`stream_petr_r50_flash_704_bs2_seq_90e`)
   as the simplest workable baseline. flash-attn will be disabled at
   inference via a head-config patch.

3. **`latent_ops.py`** ([stages/04_streampetr_latent/latent_ops.py](stages/04_streampetr_latent/latent_ops.py)):
   - `MemoryState` dataclass typing the 5 memory tensors + a `frame`
     field tracking whether positions are in `"global"` or `"ego"` coords.
   - `transform_points(points, matrix)` matches StreamPETR's
     `misc.transform_reference_points(reverse=False, translation=True)`
     bit-for-bit.
   - `warp_memory_to_ego` and `warp_memory_to_global` mirror the head's
     pre_update / post_update warp logic.
   - `step_memory(state_t, ego_pose_t, ego_pose_inv_tp1, dt)` —
     frame-to-frame helper for the future speculator.
   - **CPU self-tests pass**: round-trip identity, ego translation
     inverts in ego frame, zero-motion preserves positions while
     advancing time.

4. **`dump_latents.py`** ([stages/04_streampetr_latent/dump_latents.py](stages/04_streampetr_latent/dump_latents.py)):
   skeleton for Phase 1.1. Loads StreamPETR with the R50 checkpoint,
   iterates a scene's frames in order, snapshots the 5 memory tensors
   in their *post-update* state (= global frame) after each forward,
   saves as `.pt` per scene. Not yet runnable — needs env + data prep.

## What hit issues (and was resolved this session)

- **Initial assumption of dense BEV `[B, C, H, W]`** turned out to be
  wrong for StreamPETR. Resolved by reading the head code and switching
  the representation to sparse memory tokens. User confirmed Option A
  before any model surgery happened.
- **mmcv version incompatibility** (existing `sensor` env on mmcv 2.x vs
  StreamPETR's mmcv 1.x). Resolved by creating a parallel `streampetr_env`
  rather than porting / downgrading.

## Where we paused

Waiting on the user to set up `streampetr_env` on a GPU node and run the
four shell scripts in order. CPU-only prep (the warp utility + dumper
skeleton + this section of NOTES) is done.

### Sub-decision: val-only converter patch

Mid-session the user noticed `prepare_data.sh` was running at ~2.5 samples/s
and asked if it could be skipped. Diagnosis: StreamPETR's pkl schema
(legacy mmdet3d 1.0rc6 with `prev`/`next` sample links + 6-cam 3D→2D
projections per sample + LiDAR sweep backtracking) is genuinely
different from our mmdet3d 1.4 infos. No way to bypass without writing
~300 lines of schema-conversion code. But we only need val for Phases
0/1/2, and 28k of the 34k samples processed are train.

Resolution: surgical 2-line patch at
[StreamPETR/tools/data_converter/nuscenes_converter.py:168-173](StreamPETR/tools/data_converter/nuscenes_converter.py#L168-L173)
that does an early `continue` for train scenes. Result: ~40 min instead
of ~4 hours; train pkl is written empty. Patch is annotated with a
`PATCH (stage 04, val-only)` comment so future-us can find and revert
when train infos are needed (Phase 3 speculator training).

### Sub-decision: partial-blob tolerance patch

Re-running with the val-only patch surfaced a second issue: the cluster
has only blob 01 of 10 of nuScenes data (~3376 LiDAR samples on disk vs
34149 in metadata). The converter crashed via
`mmcv.check_file_exist(lidar_path)` on the first sample whose LiDAR
file was absent.

Resolution: replace the file-existence raise with a `continue` so
missing-file samples are silently skipped. Camera and sweep files are
packaged in the same blob as their corresponding LiDAR, so a single
LiDAR existence check is sufficient. Patch annotated as
`PATCH (stage 04, partial-blob tolerance)` at
[StreamPETR/tools/data_converter/nuscenes_converter.py:182-188](StreamPETR/tools/data_converter/nuscenes_converter.py#L182-L188).

Expected output of the converter post-both-patches: ~900 val samples in
`nuscenes2d_temporal_infos_val.pkl` (the intersection of blob-01-on-disk
and the val scene split). This matches the 914-sample subset stage 03
used.

## Plans

Phases as laid out in the user's project plan:

- **Phase 0.1** (in progress): env install + data prep + verify baseline
  inference on a single nuScenes frame produces sane detections.
- **Phase 1.1**: run `dump_latents.py` for one scene. Verify the dumps
  contain 5 memory tensors per frame with stable shapes; spot-check
  reference points are in global frame.
- **Phase 1.2**: take dumped state at t, apply `step_memory(...)`, and
  confirm `memory_reference_point` matches what the head would read into
  ego_{t+1} on the next call. This is the round-trip check that proves
  `latent_ops.py` is consistent with StreamPETR's native warp.
- **Phase 2**: simulate camera-leads / LiDAR-lags. StreamPETR is
  camera-only so the delayed-sensor story is conceptual at this stage —
  simulated by reusing stale image features for one frame while a fresh
  pose update propagates memory. Validates the abstraction before a
  second modality enters.
- **Phase 3**: the speculator. Lightweight model:
  input = `(memory_state_t, camera_features_{t+1})`,
  output = `Δmemory_embedding_{t+1}` (residual).
  Train against the difference between StreamPETR's actual
  `memory_embedding_{t+1}` and the position-only-warped baseline.

## Things to NOT do (Stage 04)

- Don't introduce a dense BEV step "to match the original plan above".
  The original plan was written assuming BEVFusion-style features; we
  now know StreamPETR is sparse and matching the math to the model is
  cleaner.
- Don't decode to boxes during speculation. Latent verification first.
- Don't try to make StreamPETR run in the `sensor` env (mmcv 2.x port
  is more work than a parallel env, and we lose stage 03 reproducibility).
- Don't dump every layer's intermediate features. The 5 memory tensors
  are the persistent state; everything else is reproducible from them
  + camera input. Keep dumps small.

## Phase 0.1 sign-off + a rabbit hole worth recording

**Status: Phase 0.1 passes.** Verified by running `tools/test.py` and
comparing the saved `results_nusc.json` scores per sample token against
the smoke test's output. They match bit-for-bit.

**The rabbit hole that ate ~2 hours:** the smoke test's `dataset[0]` is
NOT `val_pkl["infos"][0]`. `CustomNuScenesDataset` reorders samples via
`seq_split_num` + `self.flag` for streaming-inference scene continuity.
The first reordered indices happened to land on a pedestrian-heavy scene
where the model genuinely has weak per-class confidence (median pedestrian
detection_score ≈ 0.07 over the full 914-sample val).

I spent a long time debugging "why are scores so low?" when scores were
in fact identical to the official tools/test.py output for the same
sample tokens. The fix was conceptual, not code: stop comparing model
output to *expected* numbers and start comparing to *official* numbers
for the exact same sample token.

**The smoking gun** (kept for posterity):

| Token       | smoke_test top-5 scores              | tools/test.py top-5 scores             |
|-------------|--------------------------------------|----------------------------------------|
| `88a090c5…` | 0.088, 0.07, 0.063, 0.06, 0.053      | 0.088, 0.070, 0.063, 0.060, 0.053       |

All `pedestrian` class (label 8 in the StreamPETR class order:
`car, truck, construction_vehicle, bus, trailer, barrier, motorcycle,
bicycle, pedestrian, traffic_cone`). Note: I had mis-read this as
`traffic_cone` earlier in the session, which compounded the confusion.

**Lessons that carry into Phase 1+:**
- The dataset is non-trivially reordered. When dumping latents or building
  per-scene state dumps, group by `scene_token` (not by dataset index),
  and respect that the model handles scene transitions via
  `Petr3D.simple_test_pts`'s `prev_scene_token != scene_token` reset.
- For sanity checking, always pull a token out of the dataset and look
  up the same token in any reference output (results_nusc.json, etc.).
  Don't trust positional indexing.
- The patched `tools/test.py` (re-enabled single-GPU path) is now our
  authoritative reference and should stay runnable.

**Patches applied to StreamPETR (with `PATCH (stage 04)` comments for
findability):**
1. `tools/data_converter/nuscenes_converter.py:169-173` — val-only
   converter speedup.
2. `tools/data_converter/nuscenes_converter.py:182-188` — tolerate
   missing-LiDAR samples (partial blob).
3. `tools/create_data_nusc.py:66-71` — skip v1.0-test step.
4. `tools/test.py:223-225` — re-enable single-GPU test path the authors
   disabled.
5. `projects/mmdet3d_plugin/models/utils/attention.py:21-30` — guard
   flash_attn imports (no longer strictly needed since flash-attn is
   installed, but harmless and good defensive coding).

## Phase 1.2 sign-off

Re-dumped `scene_0.pt` with proper ego pose construction (the converter
saves ego2global as separate `_translation` + `_rotation` quaternion
fields; `info.get("ego2global", np.eye(4))` was silently returning
identity). Fix at
[dump_latents.py:57-69](stages/04_streampetr_latent/dump_latents.py#L57-L69)
uses `pyquaternion` to build the 4×4 from the quat + translation.

[verify_warp.py](stages/04_streampetr_latent/verify_warp.py) now reports:

| Check | Metric                                            | Value     | Status |
|-------|---------------------------------------------------|-----------|--------|
| (A)   | global → ego → global round-trip max error        | 2.4e-4 m  | OK     |
| (B)   | `ref_5[0:1024]` vs `ref_6[256:1280]` (cross-frame)| 0.0002 m  | OK     |
| (C)   | warp frame-5 global into ego_6, max BEV `|xy|`    | 68.7 m    | OK     |

`latent_ops.warp_memory_{to_ego,to_global}` now provably mirrors
StreamPETR's `transform_reference_points` on real dumped scene data.

The 1231 m FAIL on (C) the first time around was the identity-ego-pose
bug; (A) and (B) both passed under identity ego pose because (A) is
self-cancelling and (B) doesn't touch ego_pose.

# Session 2026-05-14 — Phase 2 (async sensor simulation)

## Scope decision

StreamPETR is camera-only, so "asynchronous multi-modal arrival" is
conceptual at this stage. We commit to a thin, modality-agnostic
abstraction now and validate it before a second modality enters:

- Per-camera, per-frame arrival schedule. Each `(capture frame f, cam c)`
  gets a sampled integer delay; at query time t the freshest available
  CAM_c image is the one captured at `arrival[t, c] = max{f : f + d(f,c)
  ≤ t}`. Deterministic, seeded.
- For Phase 2 we don't run the head under stale inputs (we only have
  dumps from fully-fresh runs). Instead we use the schedule abstractly
  and report the latent gap of trivial baselines between dumped states
  k frames apart — k playing the role of "how stale the freshest cam is".
- All baselines tested in *global* frame on the dumped state; the
  ego-warp itself is identity on stored ref points (Phase 1.2 (B)).
  The carried-block ref points don't drift across frames at all because
  the head freezes them in global; that's a property of StreamPETR, not
  of any speculator.

## Token correspondence — the structural insight Phase 2 made concrete

StreamPETR's head keeps memory of size `M = num_propagated + memory_len
= 256 + 1024 = 1280`. After a forward pass, `post_update`:

1. Prepends `num_propagated = 256` NEW query tokens (born at this frame
   from cross-attention with the new camera image).
2. Trims the tail so the total stays at 1280.

So between consecutive frames `f → f+1`, the correspondence is:

    f+1[256 + i]  ==  f[i]      for i in [0 .. 1023]

After k frames, the correspondence shifts by `k · 256`:

    f+k[k·256 + i]  ==  f[i]    for i in [0 .. 1280 - k·256)

For k=5 the carried block vanishes. The "new" block at the latest frame
is always `[0, 256)`, and that's the only block a learned speculator
can do anything for — the carried block is mechanical.

## Phase 2 results (one 40-frame scene, k=1..4)

[simulate_async.py](stages/04_streampetr_latent/simulate_async.py)
output (per-token L2, averaged over the scene's frame pairs):

**Carried block — mechanical baseline (sanity, expected ~0)**

| k | n_pairs | embedding mean | refpoint mean |
|---|---------|---------------:|--------------:|
| 1 | 39      | 0.0000         | 0.0001 m      |
| 2 | 38      | 0.0000         | 0.0001 m      |
| 3 | 37      | 0.0000         | 0.0001 m      |
| 4 | 36      | 0.0000         | 0.0002 m      |

Mechanically-propagated tokens are frozen-perfect across any frame
distance. Phase 3's speculator doesn't need to model them at all — it
should pass them through.

**New block — `tgt[0:256]` — embedding gap (what Phase 3 must beat)**

| k | zeros (raw L2) | copy_prev_new | copy_prev_mid |
|---|---------------:|--------------:|--------------:|
| 1 | 5.82           | **4.24**      | 4.43          |
| 2 | 5.81           | **4.39**      | 4.56          |
| 3 | 5.80           | **4.52**      | 4.69          |
| 4 | 5.79           | **4.65**      | 4.76          |

The speculator's mean-L2 target at k=1 is **< 4.24**. Beat that and the
trivial copy-the-previous-new-tokens baseline is dead.

**New block — refpoint gap (in metres, global frame)**

| k | zeros   | copy_prev_new | copy_prev_new + velo·dt |
|---|--------:|--------------:|------------------------:|
| 1 | 1292.6  | 36.15         | 36.14                   |
| 2 | 1294.8  | 38.32         | 38.31                   |
| 3 | 1297.0  | 41.28         | 41.30                   |
| 4 | 1299.0  | 44.68         | 44.73                   |

Two non-obvious findings:

1. Velocity extrapolation contributes **nothing** (36.15 → 36.14 m).
   The 36-45 m gap is dominated by *which detection queries spawn where*,
   not by their motion between frames. Phase 3 needs an image-feature-
   conditioned predictor for new-token positions; pure motion is wrong
   tooling.
2. The new-block refpoint gap grows roughly linearly with k
   (~3 m per frame), which is consistent with new tokens scattering
   across the scene volume between frames. So the speculator has a
   shrinking budget as latency grows — useful as an external pressure
   on the latency/quality trade-off.

## Phase 2 files

- [stages/04_streampetr_latent/async_schedule.py](stages/04_streampetr_latent/async_schedule.py):
  `AsyncSchedule` dataclass + `make_schedule(T, max_delay, seed)`. CPU
  self-tests pass: max_delay=0 is fully synchronous, arrival is
  monotone-non-decreasing per cam, steady-state mean staleness matches
  the analytic expectation, generation is deterministic.
- [stages/04_streampetr_latent/simulate_async.py](stages/04_streampetr_latent/simulate_async.py):
  loads a dump, evaluates the baselines for k=1..max-k, prints the per-k
  table and saves a flat `.npz` summary. Runtime on a 40-frame scene is
  ~1 min (dominated by `torch.load` on the 57 MB dump).
- [stages/04_streampetr_latent/dumps/phase2_gaps.npz](stages/04_streampetr_latent/dumps/phase2_gaps.npz):
  the numbers above, keyed `k{k}_{kind}_{block}_{baseline}_{stat}`.

## What's NOT in Phase 2 (deferred)

- A learned speculator. That's Phase 3 — the budget table above is the
  thing it has to beat.
- Detection-head pass on speculated state. We're following the
  "latent-space verification first" rule from the verifier-history
  section.
- Multi-modal asynchrony. Camera-only is the abstraction-validation
  step. When a second modality enters, `AsyncSchedule` extends by
  adding a new column (sensor) and the per-(capture, sensor) delay
  generator stays the same.
- Running the head with stale inputs. Would need to either materialize
  per-frame image features in dumps (large) or re-run the model under
  a perturbed pipeline. Phase 3 will need *some* form of this for
  training the speculator on (state_t, stale_image_features_{t+k}) →
  state_{t+k} pairs; design decision deferred until then.

# Session 2026-05-14 (cont.) — Phase 3.1 (feature extraction contract)

## Scope reframing (user spec)

Phase 3 was originally framed as "the speculator". Before writing any
speculator, user pushed for a strictly-defined first artifact: the
**latent + feature extraction contract**, not just the dumps. Reason:
the biggest silent-bug risk in Phase 3 is misalignment between the
camera features the speculator sees and the memory tokens it's trying
to predict. If the contract isn't pinned down, training will work but
produce a speculator that learns to predict slightly-wrong things and
nobody will notice.

Contract decisions (the discussion log, not just the outcome):

1. **Pre-FPN vs post-FPN, raw vs merged.** Considered three options:
   (a) full multi-scale FPN (`~22 MB/frame fp32`), (b) post-FPN
   single-level (256ch), (c) "post-attention summary" inside head.
   Chose (b): single-scale, post-FPN, head-aligned. Rationale: the
   speculator is a *latency-correction* module, not a feature
   extractor. Pre-FPN forces it to re-learn perception. Post-attention
   ties it to camera modality. Single-scale single-level is the
   semantic sweet spot.

2. **C5-derived (deepest, 1/32 res) vs C4-derived (1/16 res).** First
   instinct was C5-derived because "deeper = more semantic". On
   second look that heuristic doesn't apply to FPN: the top-down
   pathway already pushed semantic content into level 0, so level 0
   is "semantically-merged at finer spatial res", not "low-level
   features". Phase 2 already established the residual gap is a
   *spatial spawn-location* problem, not a motion problem — finer
   spatial detail is exactly what helps. Also, "always take the
   deepest level" doesn't generalize to future modalities; "what the
   head uses" does. Chose **C4-derived, 1/16 res, position_level=0**
   (Petr3D default).

3. **Hook mechanism.** Considered (a) `forward_hook` on `img_neck`,
   (b) monkey-patch `extract_img_feat`. Chose (b): captures the
   *exact* tensor the head consumes — same level selection, same
   shape, same dtype, post `position_level` indexing. Hook on
   `img_neck` would have given a list of levels we'd then have to
   re-index manually.

## What was built

- [stages/04_streampetr_latent/dump_latents.py](stages/04_streampetr_latent/dump_latents.py)
  extended:
  - Inline **feature extraction contract** documented in the module
    docstring. Anyone re-reading this in 3 months should be able to
    reconstruct the decision without re-running the discussion.
  - `_install_feature_hook(model)` monkey-patches
    `model.module.extract_img_feat` so its return value is stashed on
    `model.module._dump_last_img_feats`. Idempotent.
  - `snapshot_memory(...)` extended to accept and store `img_feats`
    (fp16, B-squeezed).
  - `SCHEMA_VERSION = 2`; each scene record now carries the version.
- [stages/04_streampetr_latent/verify_schema.py](stages/04_streampetr_latent/verify_schema.py):
  sanity checker. Asserts per-frame tensor shapes, dtype, NaN/Inf,
  monotone timestamps, per-cam non-degeneracy, and reports
  per-frame / per-scene / projected-50-scene storage so we know what
  scale-up costs.

## Verified on `scene_0.pt`

```
Scene 265f002f: 40 frames
  schema_version = 2
  img_feats shape per frame: (6, 256, 16, 44) dtype=torch.float16
  img_feats stats (frame 0): mean=-0.1383  std=46.5062
                              min=-438.7500  max=+395.0000
  per-frame img_feats size : 2112.0 KB
  scene img_feats subtotal : 82.5 MB
  whole .pt on disk        : 137.1 MB
  projected for 50 scenes  : ≈ 4.03 GB feat-only
OK — schema contract satisfied on all scenes.
```

## Observation worth flagging to Phase 3.2 (speculator architecture)

The FPN output has **std ≈ 46.5 and range ≈ ±400**. This is raw
post-FPN, pre-anything-the-head-does. The head's cross-attention does
its own LayerNorm + position embedding before consuming these
features. **Implication for the speculator**: input normalization
(LayerNorm over the C dim, or per-feature standardization on the
training set) is not optional — feeding raw fp16 features into a
transformer block will explode the loss. Architecture spec for 3.2
must own this.

## Phase 3.1 sign-off

Contract is fixed. Future re-dumps of any nuScenes scene under this
config will produce the same shape, dtype, and pipeline location. If
anything changes (e.g. swap to R101 backbone, different image size,
new neck), `schema_version` should be bumped to 3 and a migration
note added to this section — the verifier will then flag old dumps as
out-of-contract, which is the desired failure mode.

Next: Phase 3.2 — speculator architecture spec (small transformer /
MLP mixer with input LayerNorm, target = Δe on the new-block at the
target frame, training pairs sampled under random `AsyncSchedule`).

## Phase 3.2 — speculator architecture (CPU shape test only, no training)

[stages/04_streampetr_latent/speculator.py](stages/04_streampetr_latent/speculator.py),
~210 lines, 3.4M params at default config.

### Design decisions (all documented in module docstring)

1. **Residual over copy_prev_new baseline.** The two output heads
   (`head_emb_delta`, `head_ref_delta`) are **zero-initialized**, so at
   t=0 the model produces `pred = src_new_emb` exactly — i.e., the
   Phase 2 trivial baseline at 4.24 mean L2 / 36 m. Loss can only
   decrease. This is a deliberate residual-prediction setup so we
   never start training from a regime worse than the trivial baseline.

2. **Input LayerNorm is mandatory.** Phase 3.1 measured raw FPN output
   at std≈46, range±400. Without norm the first decoder layer would
   saturate. Two LayerNorms: one over image-feature channel dim, one
   over carried-memory channel dim.

3. **Camera spatial PE = learnable, per-cam.** `[N_cams=6,
   H*W=704, D=256]` parameter (≈1M params, the dominant
   non-decoder block). Without PE the cross-attention can't tell
   CAM_FRONT from CAM_BACK and can't tell near pixels from far ones.

4. **Carried-memory PE = refpoint MLP.** The carried embedding has
   no intrinsic spatial content — StreamPETR's head injects 3D
   position via cross-attention's PE inside the forward call. We
   mirror that with a small `3 → D/2 → D` MLP on the global ref
   point.

5. **Staleness embedding.** `Embedding(max_staleness+1, D)` added
   per-cam after spatial PE. So the cross-attention can implicitly
   down-weight stale cams without a hard-coded mask. This is what
   lets the speculator be schedule-agnostic at training time.

6. **Architecture size.** 2 decoder layers, D=256, 8 heads, FFN×4.
   3.4M parameters total — ~10% of StreamPETR head's size. We are
   NOT competing for capacity; we are learning a correction on top
   of a near-correct baseline.

### What the unit test verifies

Running `python speculator.py` on CPU asserts:

* Output shapes: `pred_emb=[B, 256, 256]`, `pred_ref=[B, 256, 3]`.
* Zero-delta init exactly reproduces the copy_prev_new baseline:
  `pred_emb == src_new_emb` and `pred_ref == src_new_ref` to 1e-6.
* Perturbing the delta heads breaks the identity (mean |pred - base|
  jumps from 0 to ~1.5 with `weight ~ N(0, 0.01)`).
* Backward pass propagates through all 53 trainable tensors.

```
Speculator parameters: 3,422,979 (~3.42M)
[OK] output shapes
[OK] zero-delta init reproduces copy_prev_new baseline exactly
[OK] after delta-head perturbation, mean |pred - baseline| = 1.5214
[OK] backward pass: 53 tensors received gradients
```

### Deferred to Phase 3.3 (training data pipeline)

* **Refpoint coordinate centering.** Global coords are ~1000s of m;
  the `ref_pe_mlp` cannot train stably on that. Data prep must
  subtract `ego_pose_translation` before feeding refs into the
  speculator (and add it back when comparing predictions against
  global-frame ground truth).
* **AsyncSchedule integration.** At each batch we sample a schedule,
  substitute "stale" cam features from the appropriate older
  dumped frame, and feed the staleness tensor accordingly. With
  schema_version=2 dumps this is just slicing — no model re-run
  needed.
* **Dataset scale-up.** scene_0.pt is enough for shape testing
  but ~80 MB feature payload per scene → need at least 20-50
  scenes for meaningful training. Will require one GPU pass to
  re-run dump_latents.py over the val set.

## Phase 3.3 — training pipeline + the cross-scene generalization wall

### What we built and ran

- [stages/04_streampetr_latent/dataset.py](stages/04_streampetr_latent/dataset.py):
  `SpeculatorDataset` with `(src, tgt)` pair enumeration, per-cam
  staleness sampling, ego-centered + scaled refs (REF_SCALE=50m), and
  `make_train_val_split(..., n_val_scenes=N)` for disjoint scene-level
  splits.
- [stages/04_streampetr_latent/train.py](stages/04_streampetr_latent/train.py):
  Adam + MSE loop. Reports per-token L2 (Phase 2-comparable). Flags:
  `--k-min`, `--k-max`, `--schedule-mode`, `--max-staleness`,
  `--dropout`, `--val-scenes`, `--zero-carried`, `--grad-clip`.
- [stages/04_streampetr_latent/dumps/scenes_all.pt](stages/04_streampetr_latent/dumps/scenes_all.pt):
  23 scenes from nuScenes val set, schema_version=2, ~3.1 GB on disk.

### The four runs and what they told us

All k=1, fresh schedule, MSE loss, REF_SCALE=50, batch=32, lr=1e-3.
Reported emb_l2 = mean per-token L2 on memory_embedding's new block.
Baseline = copy_prev_new = `src.memory_embedding[:256]`.

| run | scenes | dropout | steps | train emb (final) | val emb (final) | val vs baseline |
|-----|--------|---------|-------|-------------------|-----------------|-----------------|
| A: single-scene overfit       | 1  | 0.0 | 5000  | 1.22 | — (no val) | — |
| B: 8 scenes, no val           | 8  | 0.0 | 10000 | 2.25 | — | — |
| C: 8 scenes, 2-scene val      | 8  | 0.0 | 10000 | 1.99 | **6.27** | **+27%** (worse) |
| D: 23 scenes, 5-scene val     | 23 | 0.1 | ~10300 | 3.51 | **5.00** | **+8%** (worse, but milder) |
| E: 23 scenes, val, **zero-carried** | 23 | 0.1 | ~4150 | 4.12 | **4.93** | **+7%** (worse) |

**The signal in the data:**
* (A) and (B) — pure overfit on small data — work as expected.
  Architecture has capacity, training pipeline is sound.
* (C) reveals the wall: val emb climbs *monotonically* from 4.92 →
  6.27. Each gradient step on train made val worse. Anti-learning.
* (D) — 3× more train scenes + dropout=0.1 — softened the divergence
  (val gap +8% instead of +27%, train delta -1.33 vs val delta +0.39).
  Genuine progress but trend direction *unchanged*: val still grows.
* (E) is the critical diagnostic. Zeroing `carried_emb` and
  `carried_ref` before they reach the speculator (so the model can
  only use `img_feats`) gave us the **same val-growth pattern**.
  Therefore the cross-scene contamination is NOT specifically about
  the carried-memory path. img_feats are also scene-specific enough
  to memorize.

### Conceptual finding (the part to highlight in the paper)

> Raw StreamPETR memory embeddings exhibited poor cross-scene
> transferability, suggesting that the memory state acts more like
> an internal transformer computation cache than a globally stable
> predictive latent space.

What we are predicting (the new-block tokens at frame t+1) is the
output of StreamPETR's 6-layer decoder operating on the queries +
KV stack at that frame. The function is deterministic and learnable
in principle, but it is essentially a slice of the head's internal
machinery. A small (2-layer, 3.4M-param) speculator cannot reproduce
that function from outside, and the function's outputs are highly
scene-conditional rather than scene-invariant. So scene-level
held-out evaluation is the wrong yardstick — it asks the speculator
to be a scene-invariant approximator of a scene-conditional cache.

### Why we are not trying B or C from the previous strategy options

**Option B (predict head's residual on stale vs fresh inputs).**
Looked attractive mathematically (lower-variance target). But the
training signal would push the speculator to chase the head's
artifact-level quirks rather than the conceptual question. The
systems story (speculative latency hiding) gets watered down by
"learned residual on a detector that already runs". Pragmatically,
this becomes engineering fix, not contribution.

**Option C (massively more data — process train split).** The val
divergence is *monotonic* even with dropout, not "noisy near zero".
That signature is characteristic of a representation mismatch
(target not learnable from inputs under current architecture),
not insufficient training data. More data delays the divergence,
it doesn't reverse it. Bad ROI on multi-day GPU runs.

### What Phase 3 still successfully delivered

Even with the cross-scene wall, Phase 3 produced concrete artifacts
that survive the reframing:

1. Async sensor schedule formalism + generator (Phase 2's
   `AsyncSchedule`).
2. Schema-versioned latent + feature dumps with strict extraction
   contract (Phase 3.1).
3. Lightweight speculator architecture with zero-init residual heads
   and verified gradient flow (Phase 3.2).
4. Token-correspondence analysis (carried block is mechanical, new
   block is what's worth predicting).
5. The diagnostic finding above: explicit characterization of where
   raw memory embeddings break as a predictive latent.

# Session 2026-05-14 (cont.) — Phase 3.4 (pivot: within-scene latency robustness)

## Why a pivot, not a fallback

The cross-scene test was the strictest possible generalization claim
("speculator predicts new tokens for arbitrary unseen scenes from
trained weights"). It is harder than the contribution requires.
Autonomous-driving deployment looks more like:

* The vehicle operates in a known geographical region (mapped roads,
  rehearsed test areas, geofenced commercial zones).
* Sensor latencies vary frame-to-frame within that region.
* Environment priors are huge and stable per region.

Under that frame the relevant claim is:

> Given a perception model deployed in a known operational
> distribution, can a small speculator hide sensor arrival latency
> with bounded perception degradation?

This is the contribution. The cross-scene experiment was a stretch
goal we (in retrospect) attempted before validating the in-domain
claim. We now pivot to validating the in-domain claim directly.

## The new framing

* **NOT**: "general future latent prediction across arbitrary scenes."
* **YES**: "speculative continuation under bounded operational
  distribution."

This phrasing is realistic, systems-oriented, deployment-grounded
— and AV literature accepts domain-bounded assumptions as
standard.

## Phase 3.4 plan (work to be done)

1. **Frame-level train/val split** within scenes. E.g., for each
   scene of length T, train on frames `[0, T·0.75)`, hold out
   `[T·0.75, T)` as in-scene val. New dataset flag, no architecture
   change.
2. **Train under random schedules**: `schedule_mode=random`,
   `max_staleness=1` (or wider). Speculator now sees the
   "some-cams-stale" regime it was always meant for.
3. **Two-axis evaluation** on the held-out frames:
   - **In-distribution latency**: same `max_staleness=1` as train.
     Establishes that the speculator beats Phase 2 baselines on
     unseen frames within trained scenes.
   - **Out-of-distribution latency**: `max_staleness=2`. Tests
     graceful degradation: how much does perception quality drop
     when actual latency exceeds the trained envelope?
4. **Per-staleness breakdown** in eval: report emb_l2 / ref_l2 for
   each (k, staleness-vector-summary) bucket. Reviewer-relevant: the
   "latency vs quality trade-off curve" that the paper claim rests on.
5. **Phase 2 baseline on the same val pairs.** Required for honest
   comparison; the existing `simulate_async.py` numbers are scene_0
   only.

### What stays the same

* Speculator architecture, hyperparameters, REF_SCALE, MSE loss.
  All the Phase 3.2 + 3.3 infrastructure works under the new split.
* The dumps (`scenes_all.pt`) are unchanged — schema_version=2
  remains the contract.
* Phase 2's framing — carried block mechanical, new block to be
  predicted — still applies. Just evaluated with frame-level val.

### What we drop

* The cross-scene held-out claim. Documented above as a finding,
  not a failure mode of the pivot.

# Session 2026-05-15 — Phase 3.4 diagnostic: stale-input mAP degradation

## What we measured (and why this was the right thing to measure first)

Before committing to *any* mitigation pivot (latent prediction →
selective recomputation → stale-aware gating), we needed one piece of
empirical evidence we had never collected: **does StreamPETR actually
degrade under stale camera inputs?** All prior Phase 2/3 work simulated
staleness only in the speculator's input space (substituting `img_feats`
inside a detached training pipeline). We had no idea how the full head
behaves when fed stale features through the real inference graph.

If StreamPETR were already robust out of the box, there would be no
problem to solve — `Pivot 2` (stale-aware gating) would be unmotivated.

## How we measured it

[stages/04_streampetr_latent/measure_stale_map.py](stages/04_streampetr_latent/measure_stale_map.py)

* Monkey-patches `Petr3D.extract_img_feat` to maintain a per-cam ring
  buffer of recent post-FPN features. At each frame, after computing
  the fresh features, per-cam slots are overwritten with
  `cache[c][frame_in_scene - delay_c]`. Cache resets at scene
  boundaries; first frame of every scene falls back to fresh (cache
  shorter than the configured delay).
* Runs the full StreamPETR pipeline through `single_gpu_test` and then
  `dataset.evaluate` for nuScenes-devkit mAP / NDS.
* Adds a `load_gt` patch (`_install_partial_blob_eval_patch`) to filter
  GT to the 914 sample tokens our partial-blob val set actually covers.
  Required because nuScenes-devkit's
  `set(pred.sample_tokens) == set(gt.sample_tokens)` assertion compares
  against the full 150-scene val split otherwise.

Configurations: uniform per-cam delay = `0, 1, 2, 3` frames (frame
spacing ≈ 500 ms at nuScenes's 2 Hz).

## Result — large, monotone degradation

> **SUPERSEDED for delay > 0 — see the 2026-07-17 session.** These runs fed
> stale image features with the CURRENT frame's pose, calibration and an age-0
> timestamp, so they price *unlabelled* staleness (a late frame the system is
> falsely told is current), not staleness itself. They remain a valid
> measurement of the naive-deployment condition; they are not a measurement of
> information loss. The honest-protocol numbers are
> 0.4448 / 0.1849 / 0.1007 / 0.0731 for delay 0/1/2/3.
> The delay-0 row (0.4448) is unaffected — no staleness, no lie.

| delay (frames) | wall time | mAP    | NDS    | mATE (m) | mAVE   | mAOE  |
|----------------|-----------|--------|--------|----------|--------|-------|
| 0 (fresh)      | 0 ms      | 0.4448 | 0.5505 | 0.587    | 0.358  | 0.402 |
| 1              | 500 ms    | 0.1331 | 0.3090 | 1.088    | 0.418  | 0.759 |
| 2              | 1000 ms   | 0.0774 | 0.2273 | 1.025    | 0.540  | 0.969 |
| 3              | 1500 ms   | 0.0610 | 0.1955 | 0.975    | 0.714  | 0.843 |

Relative drops from fresh baseline: −70%, −83%, −86% on mAP;
−44%, −59%, −65% on NDS.

The baseline value `mAP=0.4448` for `delay=0` is slightly *above* the
official `43.2` / `53.7` headline numbers because we evaluate on the
23-scene partial-blob subset of val, which happens to be easier than
the full 150-scene val. The DEGRADATION curve, not the absolute
baseline, is what matters here.

## Per-class signature — motion-sensitive collapse

Per-class AP at delay 0 → 1 → 2 → 3:

> **SUPERSEDED for delay > 0** (same coordinate-lie confound as the table
> above). The qualitative signature — fast movers collapse first, static
> classes degrade gently — survives the honest protocol, but the magnitudes do
> not. Pending re-measurement.

| class               | delay 0 | delay 1 | delay 2 | delay 3 |
|---------------------|---------|---------|---------|---------|
| motorcycle          | 0.273   | 0.021   | 0.002   | 0.000   |
| bicycle             | 0.390   | 0.041   | 0.010   | 0.002   |
| trailer             | 0.510   | 0.000   | 0.000   | 0.000   |
| pedestrian          | 0.491   | 0.137   | 0.052   | 0.026   |
| car                 | 0.640   | 0.178   | 0.104   | 0.075   |
| bus                 | 0.380   | 0.141   | 0.082   | 0.062   |
| truck               | 0.372   | 0.066   | 0.018   | 0.018   |
| construction_veh.   | 0.265   | 0.156   | 0.079   | 0.060   |
| barrier             | 0.401   | 0.191   | 0.127   | 0.098   |
| traffic_cone        | 0.726   | 0.400   | 0.300   | 0.267   |

**Pattern**: fast-moving classes (motorcycle, bicycle, pedestrian,
trailer-as-trucked-cargo) collapse to near-zero AP within one frame of
staleness. Static or slow classes (traffic_cone, construction_vehicle,
barrier) degrade gradually, retaining 30-50% of their fresh AP at
delay=3.

This matches the intuition: stale features show objects at *past*
positions. The head still tries to localize objects in 3D, but it does
so using outdated visual evidence. For fast movers, "past position"
is far from "current ground-truth position" — detections miss by enough
that they don't match GT at any IoU threshold.

For mATE: translation error nearly doubles (0.59 → 1.09 m) at
delay=1. For mAVE: velocity error grows steadily because the temporal
context propagates corrupted state forward.

## Decision

The decision matrix from before was: `delay=1 mAP < 0.38 → strong
degradation, Pivot 2 contribution very strong`. We measured 0.13.

**Pivot 2 (stale-aware gating) is now empirically motivated.** There is
a real, large, measurable problem. A successful mitigation that
recovers any meaningful fraction of the 31-point mAP drop is a
publishable contribution.

The framing the paper now writes itself around:

> Streaming camera-only 3D detection (StreamPETR) catastrophically
> fails under modest sensor staleness — uniform 500 ms per-camera
> latency collapses mAP from 0.44 to 0.13 (−70%) on the nuScenes val
> subset. The failure is concentrated in motion-sensitive classes
> (motorcycle, bicycle, pedestrian fall to near-zero AP). We propose
> a stale-aware mitigation that suppresses unreliable new memory
> tokens when cameras are stale, allowing the carried memory (proven
> mechanically stable in our Phase 1/2 analysis) to dominate the
> downstream detection.

## Next step — measure the simplest possible mitigation

Before training a learned `α(k)` gate, we want to know: does any
naive override of the new memory block at stale frames recover mAP?
If even a hard policy (replace new block with zeros, or freeze it at
the last fresh frame's value) helps, the gating hypothesis is on the
right track. If hard policies make things worse, we need to think
more carefully about gating semantics.

Implementation: [stages/04_streampetr_latent/measure_stale_map_mitigation.py](stages/04_streampetr_latent/measure_stale_map_mitigation.py)
extends the diagnostic with a `--mitigation {none, zero, freeze}` flag.

* `none` reproduces the diagnostic (no override).
* `zero` overrides `memory_embedding[:, :256, :]` with zeros after
  `post_update_memory`, so the head's stale-derived new tokens are
  flushed before being passed to the detection head and to the next
  frame's pre_update.
* `freeze` overrides with the new-block from the most recent
  fresh frame in the same scene (cached on entry).

The override happens AFTER the head's `post_update_memory` completes,
so we don't fight the head's internal state machine. From the head's
point of view it ran normally; we just edit the persistent memory
between calls.

The same `FeatureSubstitutor` provides the staleness state — its
`was_stale` flag tells the mitigator whether the current frame used
stale features (which is when overrides apply).

## Phase 3.4 — mitigation results, round 1: zero / freeze BOTH worsen mAP

> **SUPERSEDED — measured against a contaminated baseline.** Every cell here is
> under the coordinate lie, so both the mitigations AND the `none` column they
> are compared against price unlabelled staleness. The *direction* survives with
> scene-level statistics (zero −0.038 at 0/23 scenes, freeze −0.023 at 2/23,
> both p=0.000) but the numbers are pending re-measurement on top of the honest
> key-side protocol.

| delay | none (baseline) | zero  | freeze |
|-------|-----------------|-------|--------|
| 0     | **0.4448**      | —     | —      |
| 1     | **0.1331**      | 0.0715 | 0.1008 |
| 2     | **0.0774**      | 0.0434 | 0.0615 |
| 3     | **0.0610**      | 0.0336 | 0.0454 |

In every (delay, metric) cell: `none > freeze > zero`. The naive hard
overrides we expected to mitigate damage made the model *worse* than
just letting the head's stale-derived output stand. This is a strong,
unambiguous negative result for the "suppress unreliable tokens"
hypothesis as posed.

### Why naive suppression fails — the structural insight

The stale-derived new tokens are NOT garbage. They are detections that
the head computed using features from time `t-k` — temporally lagged but
spatially structured. mATE = 1.09 m at delay=1 (no mitigation) confirms
this: detections are off by ~1 m, not random.

Equally important — and not obvious before this experiment — is what
happens to the *carried* block over time when we destroy the new block:

* At frame `f`, the head's `post_update` prepends the new block and
  the carried block becomes `[new_block_f, previous_memory[:1024]]`.
* If we zero the new block, then at frame `f+1` the head's pre_update
  reads memory whose first 256 slots are zeros. The carried block
  of frame `f+1` becomes `[zeros, new_block_f-1_as_zero, ...]` — i.e.,
  the zeros propagate FORWARD into the carried slots over time. Memory
  loses its diversity, and the head's cross-attention degrades.
* Freeze has the same problem with a different signature: the carried
  block accumulates repeated copies of the same first-frame new block.

So the head's stale-derived new tokens, despite being spatially lagged,
play a *structural* role: they keep memory diverse so that the carried
block has varied content to roll forward. Removing them is worse than
keeping them.

This is the lesson that should go into the paper:

> Naive suppression of stale-derived memory tokens degrades mAP
> further, not less. The carried memory pathway depends on each
> frame's new-block diversity to remain informative across the
> sliding window. The right mitigation must preserve the new
> block's structural role while correcting its temporal lag.

### The corrected hypothesis: temporal calibration, not suppression

Re-framing what staleness does to the head's output:

* Head reads features from time `t-k`.
* Head's cross-attention finds objects in those features → produces
  detections for the world *as it was at time t-k*.
* Head's `post_update_memory` stores those detections in global frame.
* Next call's `pre_update_memory` does ego-motion compensation using
  the current ego pose — but the OBJECT motion between `t-k` and `t`
  is not corrected by anything internal to StreamPETR.

So the head's stale-derived new block is essentially "detections at
time `t-k`, expressed in global coordinates". To re-aim them to time
`t`, we have to roll their positions forward by object velocity for
`k·Δt` seconds. StreamPETR's head already estimates this velocity
during training and writes it into `memory_velo` (Phase 1 / Phase 2
explored this tensor in latent_ops.py).

### `vel_warp` — analytical mitigation, zero new params

[stages/04_streampetr_latent/measure_stale_map_mitigation.py](stages/04_streampetr_latent/measure_stale_map_mitigation.py)
extended with `--mitigation vel_warp`:

```python
# AFTER head.post_update_memory finishes at a stale frame:
k = substitutor.last_applied_delay              # frames of staleness applied
if k > 0:
    new_ref = head.memory_reference_point[:, :256, :]
    new_velo = head.memory_velo[:, :256, :]
    new_ref[:, :, :2] += new_velo * (0.5 * k)   # nuScenes Δt = 0.5 s, 2 Hz
```

The embedding is left untouched — preserves the head's semantic /
class-confidence content. Only the xy positions are forward-projected.
z is left alone (objects don't drive vertically at the relevant
scale).

The `last_applied_delay` field on `FeatureSubstitutor` reports the
*actual* staleness applied this frame (max over cams, after the
cache-length clamp at scene start). So at frame 0 of any scene where
`last_applied_delay = 0`, `was_stale = False` and the mitigation is
skipped anyway.

### Predictions for the `vel_warp` experiment

The claim being tested:

> `vel_warp` should outperform `none` because it preserves the
> structural role of the new block (memory diversity, carried-block
> roll-forward) while correcting the dominant error mode (temporal
> position lag).

Expected pattern in the results:

* `vel_warp` improves on `none` at `delay=1` (short horizon — velocity
  estimate is reliable, distance to correct is small).
* `vel_warp` may degrade at `delay=3` (long horizon — velocity
  extrapolation accumulates error over 1.5 s, and the head's velocity
  estimate from stale inputs may itself be poor).
* If `vel_warp >> none` at delay=1, this is a "free" mitigation
  (zero learned parameters, zero training, just analytical) — strong
  deployment-friendly contribution.

If `vel_warp` also fails to recover mAP, the next move is a learned
correction: a small model that takes (stale-derived new block, k) →
corrected new block, trained on (stale-input, fresh-input) pairs from
the dumps. But analytical first.

## Phase 3.4 closure — final mitigation comparison

> **SUPERSEDED on two independent counts.**
> (1) Coordinate lie: every cell is under the naive harness (2026-07-17 session).
> (2) Statistics: `vel_warp`'s apparent +3–5% is **not significant** — paired
> across the 23 scenes it is +0.0015, 95% CI [−0.0005, +0.0037], winning 13/23
> scenes, sign p=0.68. The claim below that vel_warp is "a small, consistent
> improvement" and the paper's "one surviving positive result" are **retracted**.
> The gating conclusion (no value over pure projection) is unaffected — both
> were noise.

| delay | none (baseline) | zero   | freeze | vel_warp | vel_warp_gated |
|-------|-----------------|--------|--------|----------|----------------|
| 0     | **0.4448**      | —      | —      | —        | —              |
| 1     | 0.1331          | 0.0715 | 0.1008 | **0.1378** | 0.1372       |
| 2     | 0.0774          | 0.0434 | 0.0615 | **0.0809** | 0.0807       |
| 3     | 0.0610          | 0.0336 | 0.0454 | **0.0633** | 0.0634       |

`vel_warp_gated` with `vel_threshold = 1.0 m/s` was essentially
indistinguishable from `vel_warp` (≤ 0.001 mAP difference everywhere).
This suggests the head's velocity estimates on truly static objects
are already close to zero, so multiplying by zero (or by a small
non-zero) gives the same result. **Static gating provided no
additional value over pure analytical projection.**

The conclusion from the full analytical sweep:

* Naive suppression mitigations (`zero`, `freeze`) are STRICTLY WORSE
  than doing nothing. Stale-derived tokens carry necessary structural
  diversity for the carried-block roll-forward.
* Analytical velocity-based forward projection (`vel_warp`) provides
  a small, consistent improvement (+3-5% mAP across delays) — proves
  the direction is correct but the magnitude is far short of what's
  needed to recover the 31-point baseline drop.
* Gating doesn't help because the velocity field is already
  near-zero where it should be.

This **closes the analytical-only investigation**. The remaining gap
(0.14 vs 0.44 mAP at delay=1) must be closed with a *learned* model
if at all.

# Session 2026-05-15 — Phase 4 (reframe to proper speculation)

## Why this is a reframing, not a new direction

A fair concern raised in this session: are we still doing "speculative
asynchronous perception", or did we drift to "robust mitigation"?

The Phase 3.4 mitigation experiments are *robustness* (run head on
stale features, correct its output). Speculation has a different
shape: emit a prediction BEFORE fresh data arrives, verify when it
does, accept or rollback.

Phase 4 reinstates speculation by changing what the learned model is
FOR, not what it does mechanically:

* **Phase 3.4 view**: speculator = correction module applied to the
  head's stale output.
* **Phase 4 view**: speculator = mechanism that EMITS a detection
  before the fresh forward completes. The head's stale-derived
  output is then a *verification reference*: when fresh data arrives,
  the head runs fresh and the system can compare the speculative
  emission against it (acceptance or rollback).

Same model, same training pipeline, different role in the runtime.
This brings back acceptance rate, latency saved, and the rollback
semantics that motivated the project originally.

## Streaming runtime semantics

```
Frame t (with stale cam features available):
    speculative_detection = head(stale) → LTC_speculator → emit
    [downstream consumer can act on speculative_detection immediately]

Frame t' = t + k (fresh data finally arrives for time t):
    verified_detection = head(fresh)
    similarity = compare(speculative_detection, verified_detection)
    if similarity > τ:
        ACCEPT — the early emission was good enough
        latency saved: k · 500 ms
    else:
        ROLLBACK — keep verified_detection as authoritative
        latency: same as no-speculation
```

The paper-grade metrics naturally fall out:
* `speculation_success_rate(τ, k)` — acceptance probability at
  staleness k for threshold τ.
* `mAP_speculative` vs `mAP_verified` — quality of the early emission
  vs the eventual ground-truth-resolution.
* `effective_latency` — weighted average of (saved-k-frames if
  accepted) / (no-saving if rolled back).
* Latency–quality trade-off curve (sweep τ).

## What changes in code vs Phase 3.4

| Component | Phase 3.4 | Phase 4 |
|-----------|-----------|---------|
| `FeatureSubstitutor` | Same — defines staleness | Same |
| `MemoryMitigator` | Hard analytical override (`zero`/`freeze`/`vel_warp`) | Calls trained `LTC` |
| `LTC` (new) | — | Learned residual correction module |
| `Verifier` (new) | — | Threshold or learned classifier comparing spec vs fresh outputs |
| Output of pipeline | Single detection stream (mitigated) | TWO streams: speculative + verified |
| Eval | `mAP@delay` table | `mAP_spec`, `mAP_verified`, `accept_rate`, latency curve |

## Data scope reality

Cluster currently has only `v1.0-trainval01_blobs.tar` extracted →
3376 LIDAR samples. Breakdown:
* 914 already in val pkl (23 scenes — what we used in Phase 3.4).
* ~2462 are train samples (~60 train scenes) — but
  `nuscenes2d_temporal_infos_train.pkl` is empty (our `val-only`
  converter patch wrote it empty).

For full scale-up we have two paths:
* **Path A (work with blob 01)**: regenerate train pkl WITHOUT the
  val-only patch, get ~60 train scenes + 23 val scenes. Substantial
  upgrade vs current 23-scene-only experiments, but not "full
  nuScenes-680-scenes" claim.
* **Path B (download remaining blobs)**: ~40 GB each, manual
  download. Eventually unlocks full ~600 train + 150 val scenes.
  User decision; coursework can proceed under Path A and discuss
  Path B as future scaling.

## Pilot before full scale-up

User-requested instinct: verify the LTC training approach can possibly
work on a small subset BEFORE committing 30+ GPU-hours to full data
generation. Pilot scope on existing 23-scene val:

1. **`dump_ltc_pairs.py`** — for each scene, walk frames twice (once
   fresh, once stale-k for k∈{1,2,3}). Save per-frame tuples:
   - inputs: `stale_new_emb`, `stale_new_ref`, `stale_new_velo`,
     `carried_emb`, `carried_ref`, `k`
   - targets: `fresh_new_emb`, `fresh_new_ref`
2. **`ltc.py`** — small transformer (~1-3M params). Inputs above,
   outputs (Δemb, Δref). Initialised so output equals
   `vel_warp` analytical baseline at step 0 (zero-init delta heads +
   add the analytical correction in forward). Loss can only go down
   from the proven analytical floor.
3. **`train_ltc.py`** — Adam, MSE loss, scene-level split (e.g.,
   18 train / 5 val of the 23 val scenes). Random k sampling per
   batch.
4. **Integrate into `measure_stale_map_mitigation.py`** with
   `--mitigation ltc --ckpt <path>`. Run the same delay sweep and
   compare against analytical mitigation on full nuScenes mAP/NDS.

Pilot success criterion: LTC mAP at delay=1 ≥ 0.20 (a +0.06 gain
over `vel_warp`'s 0.138). If we hit that on the 23-scene val even
with disjoint train/val split, the architecture works and scale-up
is justified. If we get ≤ 0.14 (no meaningful gain over analytical),
diagnose why before burning GPU on full data.

Pilot GPU cost estimate: ~2-4 hours total (4 stale-config dumps × ~1
hour each, train ~30 min, eval ~5 min × few configs).

## Phase 4 pilot — executed; negative result with a precise diagnosis

The pilot ran end-to-end on the 23-scene val set. Results:

* **Training validates the architecture.** Init emb_l2 = 4.96 train /
  4.74 val (= vel_warp baseline). After 5000 steps: train = 3.95,
  val = 4.00. Both reduce together. No divergence between train and
  val emb_l2 — unlike Phase 3.3, the residual prediction task
  generalizes cross-scene. Per-k val breakdown at end: emb_l2 = 3.95
  / 4.00 / 4.04 for k=1/2/3. Model generalizes across staleness.

* **But trained LTC HURTS mAP.** Plugging the trained checkpoint into
  `measure_stale_map_mitigation.py --mitigation ltc` gives:

  | delay | none   | vel_warp | LTC (pilot) |
  |-------|--------|----------|-------------|
  | 1     | 0.1331 | 0.1378   | **0.1081**  |
  | 2     | 0.0774 | 0.0809   | **0.0611**  |

  All TP errors (mATE, mAVE, mAOE) also worse with LTC than with
  vel_warp. The model reduces per-token L2 to fresh-trajectory
  targets, but those L2 reductions DO NOT translate to mAP gain.

### Why — the chimera memory problem

The training pairs are generated by walking each scene TWICE:
* Once entirely with FRESH features → produces fresh-trajectory
  states $(F_0, F_1, \ldots)$. We save $F_t[0{:}256]$ as the target.
* Once entirely with STALE features (uniform k) → produces stale-
  trajectory states $(S_0, S_1, \ldots)$. We save $S_t[0{:}256]$ and
  $S_t[256{:}1280]$ as the inputs.

So the target $F_t[0{:}256]$ belongs to a memory state where the
carried slots $F_t[256{:}1280]$ were also produced by fresh forward
passes — i.e., the target lives in a fresh-trajectory context. The
input carried slots $S_t[256{:}1280]$ belong to a stale-trajectory
context.

At inference we always-stale. At frame $t$ the head produces $S_t$,
and we ask LTC to predict "what would the new block look like in
fresh trajectory?". When we then write that prediction back into
`head.memory_embedding[0{:}256]`, memory becomes:

```
slot [0:256]      = LTC prediction      (≈ fresh-trajectory shape)
slot [256:1280]   = stale-trajectory's carried   (from S_t)
```

This is a *chimera*: slots from two different sample trajectories
splice into one tensor. The head's cross-attention was trained on
self-consistent memory (all slots from the same forward sequence).
With a chimera state, downstream behavior is unpredictable — and
in this run, worse than just leaving the head's stale-derived
output alone.

### What this proves (the substantial finding)

> Optimizing L2 distance in StreamPETR's memory latent space to a
> fresh-trajectory target produces consistent, generalizable L2
> reductions on held-out scenes — but does NOT translate to mAP
> recovery. The reason is structural: the memory at inference time
> is in a stale-trajectory regime, and any prediction tuned toward
> fresh-trajectory targets produces a chimera state that the
> detection head's cross-attention was not trained to handle.

This is a *rich* negative result. It rules out the entire family of
"latent prediction + L2 supervision" approaches for this problem,
and points concretely to what would be needed to succeed.

### Comparison to Phase 3.3 negative

Phase 3.3 showed cross-scene generalization failure when predicting
the head's NEW tokens absolutely from carried + image. That was
about *generalization*: the target was scene-conditional, the model
memorized.

Phase 4 pilot is a DIFFERENT failure: the residual-prediction task
generalizes fine cross-scene, but the OPTIMIZATION TARGET is
mismatched with what's actually useful. L2 ≠ mAP because of
inference-time state-mixing.

### Path forward: context-matched dump (Option α)

Fix: training data must pair (stale carried + stale new) → (stale
carried + WHAT HEAD WOULD PRODUCE WITH FRESH FEATURES ON THAT SAME
STALE CARRIED).

Concretely, modify `dump_ltc_pairs.py` to, at each stale-trajectory
frame:

1. Save the head's pre-forward memory state.
2. Run the head forward with STALE features → record
   $S_t[0{:}256]$ (input).
3. Restore memory state.
4. Run the head forward with FRESH features (substituting in the
   true-fresh features for time $t$) → record the resulting
   $[0{:}256]$ slot as the TARGET.
5. Restore memory state and continue the stale trajectory.

Both runs share the same incoming memory state (which is in stale-
trajectory regime by construction). The target now lives in the SAME
context the LTC will see at inference. No chimera.

Dump cost ~2× current (two head passes per frame instead of one fresh
walk + N stale walks). For 23-scene pilot ≈ 4 min total. Training and
eval cost unchanged.

If this fix also fails to recover mAP, the conclusion would shift to:
L2 in latent space is fundamentally a poor proxy for detection
quality, and we need detection-aware loss (run head + GT matching
at training time — Option β, much heavier).

# Session 2026-07-17 — Audit of our own harness: the staleness cliff is mostly a coordinate lie

> **Read this before citing any number from Phase 3.4 or Phase 4.** This session
> did not add a mitigation. It audited the measurement apparatus underneath all
> of stage 04 and found a confound in the anchor number, plus a statistical
> error in the one result we called positive. Two prior conclusions change.
> Per lab-notebook convention the earlier sections are left intact — see
> "What this overturns" below for the links.

## Why we went looking

A council review of the six candidate paper reframings kept returning, from
three independent directions, to the same suspicion: our staleness harness
substitutes only the post-FPN image features and leaves every piece of metadata
at frame `t`. Verified in source this session (not inferred):

| site | what it does |
|------|--------------|
| [measure_stale_map.py](stages/04_streampetr_latent/measure_stale_map.py) `FeatureSubstitutor._hook` | patches ONLY `Petr3D.extract_img_feat`; `data['timestamp']`, `data['ego_pose']`, `lidar2img` all come from the CURRENT sample |
| [streampetr_head.py:402](StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L402) | `img2lidars = data['lidar2img'].inverse()` — stale pixels are lifted into the ego frame at `t-k`, then read as ego-frame-at-`t` coordinates |
| [streampetr_head.py:426](StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L426) | `rec_ego_pose = torch.eye(4)` — unconditional |
| [streampetr_head.py:438](StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L438) | `time_embedding(pos2posemb1d(torch.zeros_like(...)))` — unconditional |

So a query born from stale photons is told **"you are here, now"**. The carried
block is NOT affected: `pre_update_memory` / `post_update_memory` track true
elapsed time and pose correctly. The lie is aimed only at image-born queries.

**This is not a bug in StreamPETR.** In fresh operation `(identity, dt=0)` is
correct by definition. It is an unconditional *synchrony assumption*: the
architecture has no port through which "this observation is old" could be
expressed for image-born queries. A naive deployment wrapper around a
synchrony-trained model does exactly what our harness did — which is why the
measurement is still interesting, just not what we claimed it was.

## Experiment 1 — does the collapse scale with EGO motion? (0 GPU)

**Idea / hypothesis.** If the coordinate lie drives the collapse, degradation
must be mild where the ego barely moved and severe where it moved a lot. If
near-static frames collapse just as hard, ego motion is not the story and the
Phase 3.4 number stands as measured.

**Changes.** No inference. Every config already saved per-sample detections
under `dumps/stale_map_*/pts_bbox/results_nusc.json` (914 tokens each), so the
whole table can be re-analysed on CPU by re-running the devkit on token subsets.
New: [eval_subset.py](stages/04_streampetr_latent/eval_subset.py) (cached
NuScenes + GT, token-filtered `load_gt` patch) and
[stratify_by_egomotion.py](stages/04_streampetr_latent/stratify_by_egomotion.py).

**Setup.** commit `078e7af` · env `streampetr_env` · no W&B (CPU post-processing
of existing artifacts, no training — noting the gap rather than omitting it).
Correctness gate: `python eval_subset.py --self-test` reproduces the recorded
mAP for delay0 / delay1 / d1_velwarp to **delta 0.000000**.

Two details that matter: the first `k` frames of each scene fall back to fresh
(the substitutor clamps to cache length) and are excluded; and absolute mAP is
NOT comparable across strata (nuScenes averages AP over all 10 classes and the
strata have different class mixes), so the comparable quantity is the retention
ratio within a stratum.

**Metrics.** Retention = mAP(delay=1) / mAP(delay=0) on the SAME tokens:

| ego displacement over 500 ms | n | fresh mAP | delay=1 mAP | retention |
|------------------------------|-----|-----------|-------------|-----------|
| [0.00, 0.50) m (near-static)  | 133 | 0.3290    | 0.2823      | **85.8%** |
| [0.50, 2.00) m                | 129 | 0.4033    | 0.1696      | 42.1%     |
| [2.00, 4.00) m                | 310 | 0.4235    | 0.1377      | 32.5%     |
| [4.00, inf) m                 | 319 | 0.4989    | 0.0576      | **11.5%** |

**The confound we then killed.** Low ego motion correlates with low scene
dynamism (stopped at a light → everything else stopped too), so the table above
is on its own consistent with a boring explanation. Discriminator: classes that
*cannot move*.

| ego displacement | traffic_cone (fresh→d1) | barrier (fresh→d1) |
|------------------|-------------------------|--------------------|
| [0.00, 0.50) m   | 0.705 → 0.688 (**97.5%**) | 0.338 → 0.283 (83.6%) |
| [0.50, 2.00) m   | 0.669 → 0.315 (47.1%)   | 0.526 → 0.285 (54.3%) |
| [2.00, 4.00) m   | 0.772 → 0.341 (44.2%)   | 0.543 → 0.255 (46.9%) |
| [4.00, inf) m    | 0.797 → 0.239 (**30.1%**) | 0.535 → 0.165 (30.9%) |

A traffic cone is equally stationary in a jam and on a highway. Object motion
cannot explain a 97.5% → 30.1% swing; the only variable is how far the ego moved.

**Verdict: worked — hypothesis confirmed.** The damage is ego-induced. The
headline "500 ms staleness collapses mAP by 70%" is not a property of staleness;
it is a function of ego displacement, and it prices *unlabelled* staleness (a
frame the system is falsely told is current), not staleness itself.

Output: `dumps/stratified_egomotion_d1_fixed.json`

## Experiment 2 — are the mitigation deltas real? (0 GPU)

**Idea / hypothesis.** The effective sample size is 23 scenes, not 914 frames
(2 Hz, same objects re-observed for ~20 s → heavy within-scene correlation).
`vel_warp` (+0.005) and the LTC v1→v2 gap (+0.003) may be inside scene-to-scene
variance. Nothing in the repo had ever computed a CI.

**Changes.** [bootstrap_ci.py](stages/04_streampetr_latent/bootstrap_ci.py):
per-scene mAP for each config, cluster bootstrap over scenes (10 000 resamples)
for CIs, plus paired per-scene tests against the `delay1` baseline (exact sign
test + Wilcoxon). Bootstrapping the per-scene statistic — rather than pooled mAP
— because a resample repeats scenes and the devkit keys detections by token.
Note mean-of-per-scene mAP (0.0858) is not the pooled mAP (0.1331); the paired
tests do not depend on that identity and are the primary result.

**Setup.** commit `078e7af` · env `streampetr_env` · seed 0 · no W&B (as above).

**Metrics.** Paired per-scene, vs `delay1` baseline, n=23 scenes:

| config | mean Δ | 95% CI on Δ | wins | sign p | significant |
|--------|--------|-------------|------|--------|-------------|
| d1_velwarp | +0.0015 | [−0.0005, +0.0037] | 13/23 | 0.678 | **no** |
| d1_zero    | −0.0381 | [−0.0496, −0.0272] | 0/23  | 0.000 | yes |
| d1_freeze  | −0.0234 | [−0.0344, −0.0141] | 2/23  | 0.000 | yes |
| d1_ltc     | −0.0185 | [−0.0286, −0.0100] | 2/23  | 0.000 | yes |
| d1_ltcv2   | −0.0170 | [−0.0265, −0.0089] | 4/23  | 0.003 | yes |

**Verdict: didn't work — `vel_warp` is not distinguishable from noise.** It wins
on 13 of 23 scenes, a coin flip. The negative results, by contrast, are strong
and survive: suppression really does hurt, and LTC really does hurt.

Output: `dumps/bootstrap_ci_d1.json`

## Experiment 3 — the honest protocol (built, math validated, NOT yet run)

**Idea / hypothesis.** A real async camera stack delivers a frame captured at
`t-k` *together with* its capture timestamp and (via 100–200 Hz odometry) its
capture pose. So the honest condition is not to hide the staleness but to
**declare** it. Two zero-parameter edits using only information a real system
has:

* **Key side (the 3D lift).** `lidar2img_used = lidar2img_{t-k} @ T_{lidar(t-k)<-lidar(t)}`,
  where `T_{lidar(t-k)<-lidar(t)} = ego_pose_inv(t-k) @ ego_pose(t)`.
  NB this is *not* "swap in the `t-k` calibration": `lidar2img` is a rigid
  calibration and is ~identical at `t` and `t-k`, which is exactly *why* the
  error goes uncorrected. The correction is the composition with ego motion.
* **Query side (the conditioning).** Feed `(T_{lidar(t)<-lidar(t-k)}, dt=k*0.5)`
  instead of the hardcoded `(identity, 0)`.

The query-side edit is **in-distribution for the pretrained weights**:
`rec_ego_motion = cat([zeros(3), pose[:3,:].flatten()])` → 15 dims → the SAME
`MLN` the memory path feeds with `cat([memory_velo(2), memory_timestamp(1),
memory_egopose(12)])`. Those three zeros sit exactly in the
`[velo, velo, timestamp]` slot, so writing `[0, 0, k*0.5]` with a real relative
pose is the format the shared MLN was already trained on. Memory bookkeeping
then follows for free: `post_update_memory` stamps
`memory_egopose = ego_pose(t) @ T = ego_pose(t-k)` — literally "the pose where
these photons were captured". Only `rec_timestamp` (hardcoded 0) needs a nudge.

**Changes.** [honest_staleness.py](stages/04_streampetr_latent/honest_staleness.py)
with `--honest-mode {none,key,query,both}` so the two halves can be attributed
independently, plus
[submit_honest_staleness.sh](stages/04_streampetr_latent/submit_honest_staleness.sh)
(7-task a100 array, ~20–30 min/task).

**Setup.** commit `078e7af` · config `stream_petr_r50_flash_704_bs2_seq_90e` ·
ckpt `checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth`.

**Validation done so far (CPU, before spending GPU).** Transform algebra checked
against 200 real moving frame pairs from the val pkl:

| check | result |
|-------|--------|
| naive lift error (what runs today) | **3.335 m** mean, max 6.29 |
| corr(ego displacement, naive error) | **0.9716** |
| honest lift error (the patch) | **0.000000 m** |
| equivalence of the two formulations | residual 9.5e-13 |

i.e. the current code misplaces every stale-lifted point by almost exactly the
ego displacement, and the honest composition recovers the true position exactly.

**Verdict: WORKED — but only one of the two halves.** Results below
(job `49034769`, 7 tasks, ~3 min each on one A100).

**Gates first (both PASS, exactly):**

| gate | expected | measured | |
|------|----------|----------|---|
| `honest_d0_both` (forced no-op: T=I, dt=0) | 0.4448 | 0.4448 | PASS |
| `honest_d1_none` (patch installed, inactive) | 0.1331 | 0.1331 | PASS |

**Metrics at delay=1** (fresh = 0.4448):

| config | mAP | NDS | mATE | vs naive | % of fresh |
|--------|-----|-----|------|----------|------------|
| naive (no honest edits) | 0.1331 | 0.3090 | 1.088 | — | 29.9% |
| **honest KEY only** | **0.1849** | 0.3650 | 1.041 | **+0.0518** | **41.6%** |
| honest QUERY only | 0.0908 | 0.2353 | 1.101 | −0.0423 | 20.4% |
| honest BOTH | 0.1242 | 0.3183 | 1.066 | −0.0089 | 27.9% |

Held to the same standard that killed `vel_warp` — scene-level paired test,
n=23 (`dumps/bootstrap_ci_honest.json`):

| config | mean Δ | 95% CI on Δ | wins | sign p | significant |
|--------|--------|-------------|------|--------|-------------|
| **honest_d1_key** | **+0.0348** | [+0.0241, +0.0465] | **23/23** | 0.000 | **YES** |
| honest_d1_query | −0.0246 | [−0.0347, −0.0154] | 1/23 | 0.000 | yes (harmful) |
| honest_d1_both | +0.0028 | [−0.0126, +0.0169] | 14/23 | 0.405 | no (cancellation) |

23/23 scenes. For contrast, `vel_warp` won 13/23 (p=0.68).

**Mechanism check — the prediction made in advance, confirmed.** If this is the
coordinate lie, the fix must be INERT when the ego is stationary (T≈I) and help
in proportion to ego displacement:

| ego displacement | naive | honest KEY | gain |
|------------------|-------|------------|------|
| [0.00, 0.50) m | 85.8% | 86.9% | **+1.1 pp** (inert, as predicted) |
| [0.50, 2.00) m | 42.1% | 66.1% | +24.0 pp |
| [2.00, 4.00) m | 32.5% | 42.1% | +9.6 pp |
| [4.00, inf) m | 11.5% | **23.6%** | +12.1 pp (doubles) |

**How large is this, stated honestly.** +39% *relative* to the stale baseline
(0.1331 → 0.1849) is the flattering framing and should NOT be the headline. The
gap to fresh is 0.4448 − 0.1331 = 0.3117; the key fix closes 0.0518 of it =
**16.6% of the gap**. The remaining 83% is object motion + the spatial-spawn
problem — genuine, irreducible information loss. So: the interface artefact was
~1/6 of the cliff, and running late really does cost the other 5/6.

**Why QUERY hurts — a correction to this session's own reasoning.** I argued the
query-side edit was "in-distribution for the pretrained weights" because
`rec_ego_motion` has the same 15-dim layout the memory path feeds the shared
MLN. **That argument is wrong.** Matching the *format* of a conditioning code is
not the same as matching the *joint* distribution of (feature, condition): the
model has never seen the pair (newly-born image query, non-zero motion code) —
that slot was `(identity, 0)` in every training step. Injecting truth through a
port the model was never trained to receive it on degrades the state.

This is a **conditioning chimera**: the same structural law as the Phase 4
memory chimera, one level up. It is a third independent confirmation of the
project's central claim, and it retro-explains the LTC failure — LTC also fed a
well-formatted tensor into a pathway whose training-time joint distribution it
did not respect.

**What it does NOT fix:** object motion during the staleness gap — the 83%
residual above. Separating the ego term from the object term was the point, and
the split is now measured: **ego artefact ≈ 1/6, real information loss ≈ 5/6.**

**Superseded:** tasks 5/6 ran delay=2/3 in `both` mode, which is contaminated by
the harmful query half (0.0677 / 0.0412). Re-running as `key`-only in tasks 7/8;
until those land there is no trustworthy deep-staleness number.

## What this overturns

1. **"vel_warp recovers +3–5% mAP at zero learned-parameter cost."**
   Stated in [Phase 3.4 closure](#phase-34-closure--final-mitigation-comparison)
   and in the paper abstract/conclusion as "the one surviving positive result".
   Experiment 2 shows the delta is inside scene-to-scene variance (95% CI
   includes 0, sign p=0.68). **The paper's only positive claim must be
   retracted.** This also removes the positive evidence for the "geometric
   fields first" design principle — it survives as a hypothesis, not a result.

2. **"A single 500 ms stale frame collapses mAP by 70%."**
   True as an aggregate over this subset, but it conflates two things. Retention
   is 85.8% at near-zero ego motion and 11.5% above 4 m. The number prices the
   *synchronous input contract*, not staleness per se. Now quantified: telling
   the model where the photons were captured recovers 16.6% of the gap
   (0.1331 → 0.1849, 23/23 scenes), so ~1/6 of the published cliff was our
   measurement artefact and ~5/6 is real.

3. **"The one surviving positive result is analytical velocity projection."**
   Replaced. The surviving positive result is now the **honest key-side lift**:
   larger (+0.0518 vs +0.0015), significant where vel_warp was not (23/23 vs
   13/23 scenes), equally parameter-free, and — unlike vel_warp — it has a
   mechanism confirmed by an advance prediction (inert at zero ego motion,
   doubling retention above 4 m).

**What survives untouched:**

* `zero` = 0.072 → the 1024 carried tokens alone retain only ~16% of fresh mAP.
  StreamPETR's temporal memory is a refinement channel, not a state.
* Suppression propagates zeros forward into carried slots and drains diversity
  (now with CIs: −0.038, 0/23 scenes).
* The chimera memory problem and its context-matched fix (L2 4.07/4.07, mAP
  still 0.111). Note the LTC trained and evaluated under the SAME harness, so
  the coordinate lie is not a confound for the L2→mAP gap. It IS still exposed
  to the separate critique that no mAP-based checkpoint selection was ever done
  (`train_ltc.py` `_eval()` computes only L2) — untested, see below.
* **All of stage 05 (MamBEV).** Its cosine statistics are computed between
  honestly-computed adjacent frames and never touch this harness. The
  845-pair cache result is unaffected and is now the most robust pillar we have.

## Reframing this suggests

The harness is not broken — it is the **naive-deployment condition**. A real
stack knows the capture time and pose, but a wrapper around a synchrony-trained
model has nowhere to put that knowledge, so it does exactly what we did. That
makes this a two-arm experiment, and the gap between the arms prices the
synchronous interface contract itself.

Supporting free measurement: nuScenes cameras have a real **42.6 ms** spread
within each "synchronous" keyframe (min 42.1, max 43.8 over 200 samples), and
every method on the leaderboard — ours included — collapses all six cameras to
one pose and one timestamp. So the hazard is not hypothetical or specific to us.

Working name for the hazard: the **timestamp chimera** (interfaces silently
stamping multi-time observations as "now") — distinct from, and upstream of, the
memory chimera of Phase 4.

## Open / next

* ~~Run the array; check gates~~ DONE (both gates exact) and ~~`key` vs `query`
  vs `both` attribution~~ DONE (key helps, query hurts, both cancel).
* **In flight:** tasks 7/8 = delay 2/3 in `key` mode, to replace the
  `both`-contaminated 0.0677 / 0.0412. Expected shape if the mechanism holds:
  key-only should help MORE at deeper staleness, since ego displacement grows
  ~3 m per frame while the fix stays exact. If it does not, the ego-artefact
  story has a ceiling we have not understood.
* Untested critique worth closing: sweep LTC checkpoints every 500 steps and
  report delay=1 mAP for each. If an early checkpoint beats its init before
  degrading as L2 improves, that *earns* the word "structural" for the L2→mAP
  gap; if mAP decays monotonically from step 0, the current claim is unearned
  and a task-aware loss remains untried.
* Prior art to engage before submission: Li et al. "Towards Streaming
  Perception" (ECCV 2020), StreamYOLO (CVPR 2022), RoboBEV / Robo3D
  ("Frame Lost" ≈ our k=1), Clockwork Convnets, Deep Feature Flow, and the
  feature-imitation distillation line (Chen NeurIPS 2017; Wang CVPR 2019) where
  "L2 improves, mAP doesn't" is already known.
* Title "Latent Caches Are Not World Models" points at GAIA-1/Vista, which are
  explicit generative rollout models nobody claimed StreamPETR's memory to be.
  Strawman risk; needs a new target.
* **Paper surgery required** (not yet done — deliberately left for the author):
  the abstract and conclusion both sell vel_warp as "+3–5% mAP ... a deployable,
  parameter-free mitigation" and "the one surviving positive result". Both
  sentences are now false and must be replaced by the honest key-side result.
* Worth testing next, cheap and mechanism-driven: the conditioning chimera
  predicts that the query-side edit fails because the head was never TRAINED on
  (new query, non-zero dt). That is a falsifiable claim — fine-tune only the two
  MLN modules (`ego_pose_pe`, `ego_pose_memory`) under delay augmentation and
  see whether query-side truth flips from harmful to helpful. Small parameter
  count, fits the 1-hour job convention, and either outcome is publishable:
  it recovers more of the gap, or it confirms the chimera law holds even with
  the port explicitly trained open.

# Session 2026-07-18 — Baseline audit: MamBEV's own mAP, measured for the first time

## Why

Before trusting any degradation number we should be able to say what the
undamaged models score. Two gaps:

1. **MamBEV had never been evaluated.** Not on full val, not on our subset, not
   once. Stage 05 verified only that the checkpoint loads (1101/1105 weights)
   and that a synthetic forward on random images returns 300 boxes — then went
   straight to analysing the encoder's BEV and concluding "query-position cache,
   not a metric world grid". That conclusion rested on the internals of a model
   whose task performance was unknown. If MamBEV were broken here (pkl format,
   R101-DCN backbone, the `frames=[-2,-1,0]` stack, partial blob), the cosine
   statistics would have been measured on a non-working model.
2. **StreamPETR's +1.3 over published** had been explained in these notes as
   "our subset is easier" but never checked.

## Result — MamBEV is healthy

[measure_mambev_baseline.py](stages/06_gated_bev_psm/measure_mambev_baseline.py),
job 49074246, 12 min on one A100, 914 samples.

| model | published (full val) | ours (23-scene subset) | delta |
|-------|----------------------|------------------------|-------|
| StreamPETR R50-900q | 43.2 / 53.7 | **44.48 / 55.05** | +1.28 / +1.35 |
| MamBEV-small | 42.2 / 52.5 | **40.48 / 50.74** | −1.72 / −1.76 |

MamBEV reaches **96% of its published mAP**. Both models land within ~2 points
of their official numbers on the identical 914 samples / 23 scenes with
identical GT (token sets verified equal between the two pkls).

**Consequence: stage 05's cache result stands on a working model.** That matters
because stage 05 is the one pillar untouched by the coordinate lie — its cosine
statistics come from honestly-computed adjacent frames.

### The asymmetry, and why it is not alarming

StreamPETR lands above its published number while MamBEV lands below, so "the
subset is easier" cannot be the whole story. Per-class, it is mostly one class:

| class | instances in subset | MamBEV | StreamPETR (delay 0) |
|-------|--------------------:|-------:|---------------------:|
| **trailer** | **76** | 0.188 | 0.510 |
| motorcycle | 150 | 0.368 | 0.273 |
| bicycle | 684 | 0.442 | 0.390 |
| pedestrian | 7344 | 0.514 | 0.491 |
| car | 9608 | 0.606 | 0.640 |

nuScenes averages AP over all 10 classes equally, so trailer at n=76 moves mAP
by ~0.016 on its own — essentially the entire −0.017 gap. Small-sample variance,
not a defect. Worth noting for the paper: MamBEV is *better* than StreamPETR on
exactly the motion-sensitive classes (motorcycle +0.095, bicycle +0.052,
pedestrian +0.023) that collapse hardest under staleness.

## Config traps found (all upstream dev settings, all fatal or silent)

The config is authored for MamBEV's own training loop. Three settings had to be
overridden in `prepare_cfg()`; the upstream file was left untouched.

| setting | effect if left alone |
|---------|----------------------|
| `indices=500` on val_dataset | **silently wrong**: 500 of our 914 samples, an arbitrary prefix rather than the 23 scenes, then a devkit `set(pred)==set(gt)` failure |
| `visualizer` → `WandbVisBackend` | **crash** in `Runner.__init__` — wandb is not installed in mambev_env (this killed the first run) |
| `img_backbone.init_cfg` → `./ckpts/...` | path does not exist in this clone; redundant since `load_from` supplies every weight |

Guard added: the script aborts unless the dataset has exactly 914 samples. That
is the check whose absence would have turned `indices=500` into a neatly
formatted wrong number.

**EMA checked, no override needed.** The checkpoint carries `ema_state_dict`, so
which weights get evaluated is not obvious. Per the installed mmengine source,
`before_save_checkpoint` stores the *averaged* weights in `state_dict` ("so that
we can directly load the averaged model weights for deployment"), and the
swap-back branch of `after_load_checkpoint` requires `runner._resume`, which
`test()` never sets. Both paths therefore evaluate the same EMA weights;
`EMAHook` was left in place.

## Honest caveats

* This is a **consistency check, not a reproduction**. Nothing here shows the
  23-scene subset is representative; it shows neither model is broken. A real
  reproduction needs full val (~400 GB download for ~5 GB of camera keyframes).
* The MamBEV val pkl is **our own construction** — `enrich_val_pkl.py` added
  `scene_token`/`frame_idx` by devkit lookup, and MamBEV's temporal iterator
  depends on both for `frames=[-2,-1,0]`. Reaching 96% of published argues
  strongly that the ordering is right, but that has not been independently
  verified.
* Config parses fine on a login node with
  `Config.fromfile(..., import_custom_modules=False)` — MamBEV needs CUDA only
  for the *plugin* import. Discovering this late cost three failed submissions;
  `prepare_cfg` is now CPU-testable.

# Session 2026-07-21 — The decisive test: train the conditioning port, does the chimera dissolve?

## The question this settles

Every "state edit fails" result so far used a FROZEN, synchrony-trained model.
The strongest rebuttal to the whole thesis was therefore mundane: *of course
injecting age fails — the model was never trained to receive it.* The query-side
conditioning result (−0.032, 1/23 scenes) fed the head TRUE `(pose, dt)` in the
exact format its own MLN consumes, and it still hurt — but through a port the
frozen weights never saw during training. This session closes that rebuttal by
giving the state-editing approach every advantage: a dedicated, trainable
conditioning port, optimised end-to-end on the detection loss, under conditions
identical to evaluation.

Pre-registered success criterion: delay=1 mAP vs the honest-key baseline 0.1849.
Above → the conditioning chimera dissolves with training (engineering problem).
At/below → it does not (evidence for structural, now earned not asserted).

## What was built

- Regenerated infos WITH train scenes. The stage-04 val-only converter patch is
  now switchable (`STREAMPETR_INCLUDE_TRAIN=1`,
  [nuscenes_converter.py](StreamPETR/tools/data_converter/nuscenes_converter.py)
  `PATCH (stage 07)`); job wrote to a NEW tree
  [data_withtrain/](data_withtrain/) so `data_full/` — which every prior
  measurement depends on — is untouched. Result: **2462 train samples / 62
  scenes** + 914 val / 23, and the regenerated val token set matches the
  original pkl exactly (converter env has not drifted since May). Note 62 scenes
  is **8.9% of the official 700-scene train split** — enough for the binary
  question, not for a competitive detector claim.
- [age_port.py](stages/07_delay_conditioning/age_port.py):
  `AgeConditioningPort` — a dedicated `ego_pose_pe` / `ego_pose_memory` /
  `time_embedding` for the image-born query path, initialised as exact copies of
  the frozen head's shared modules. **Verified bit-identical at init** (max
  |Δ|=0.0 at dt=0), so fresh-frame behaviour starts exactly at the pretrained
  model and can only move where delay > 0. ~422k trainable params (1.12%).
- [train_age_port.py](stages/07_delay_conditioning/train_age_port.py):
  fine-tunes THROUGH the eval-mode test path (sequential, batch=1, memory
  evolving as at inference) with the existing FeatureSubstitutor +
  honest-key-lift hooks, optimising the head's own detection loss. Everything
  frozen but the port (0 model params require grad).

Four submissions to get running (all mechanical wiring, all now caught by a CPU
end-to-end harness): train-vs-eval feature shape (6-D queue axis), img_metas
passed positionally AND via `**data`, and `cfg.model.train_cfg` nulled (copied
from the eval scripts) so the head never built its assigner. The lesson logged:
validate plumbing on CPU up to the CUDA boundary BEFORE submitting, not the
science alone.

## Result — the port does NOT help; the chimera does not dissolve

Full run: job 49247400, 2 epochs / 4924 steps over 62 scenes, 12 min on one
A100. Training was healthy — the failure is not a broken run:

* loss ranged 4–22, tracking `k` (deeper staleness → higher loss);
* delay histogram balanced (k0=1129, k1=1225, k2=1430, k3=1136) — the port saw
  ample staleness, not a degenerate k=0 curriculum;
* all 16 port tensors moved (gradients reach every part).

| config | mAP | vs baseline |
|--------|-----|-------------|
| honest key (baseline to beat) | **0.1849** | — |
| + trained age port (4924 steps) | **0.1713** | −0.0136 |

Scene-level paired test, n=23 ([bootstrap_age_port.json](stages/04_streampetr_latent/dumps/bootstrap_age_port.json)):

| | mean Δ | 95% CI | wins | sign p | significant |
|-|--------|--------|------|--------|-------------|
| age port vs honest-key | −0.0072 | [−0.0132, **+0.0001**] | **5/23** | 0.011 | **no** (CI grazes 0) |

**Verdict: the trained port does not recover the gap.** By the pre-registered
rule the CI includes 0 (by 0.0001), so this is not a *significant harm* — but it
is unambiguously not help: 5 wins of 23, negative point estimate. The port had
the truth, the capacity, the gradient signal, and a balanced curriculum, and
still could not turn observation age into better detections.

## What this earns — and what it deliberately does NOT

**Earned.** The conditioning failure is no longer "the port was never trained."
It was trained, end-to-end, under eval-matched conditions, and did not help.
That is the strongest form of the result and closes the obvious rebuttal.

**NOT earned — "structural" is still not proven.** Three live alternatives, in
order of plausibility, to be stated by us in the paper rather than found by a
reviewer:
1. **Data** — 62 scenes / 8.9% of train. Cross-scene conditioning may need more.
2. **Capacity/placement** — 422k params at query birth; age may need to modulate
   cross-attention itself, not the query embedding.
3. **Single seed, no LR sweep** — standard hyperparameter caveats.

Unlike the LTC line, there is no train/test protocol mismatch here: the port
trains and evaluates under identical conditions (eval-mode forward, honest key
geometry, same staleness mechanism). So the LTC-style confound is ruled out;
the three above are not.

## The unified table (all interventions, clean honest-key baseline)

| intervention | site | Δ mAP @ delay1 | wins | verdict |
|--------------|------|---------------:|------|---------|
| zero new block | state | −0.056 | 0/23 | harmful |
| freeze new block | state | −0.034 | 1/23 | harmful |
| query conditioning (untrained) | state | −0.032 | 1/23 | harmful |
| **age port (trained, 422k)** | **state** | **−0.007** | **5/23** | **no help** |
| vel_warp | state | −0.000 | 13/23 | noise |
| **honest key-side lift** | **INPUT** | **+0.052** | **23/23** | **works** |

Five ways to edit or condition the latent state — none help. One input-side
geometric correction — works in every scene. The trained port is what makes this
credible: it is the case where the state-editing approach was given every
advantage and still lost. This is the cleanest statement of the project's law:
**you cannot fix a streaming detector's staleness by editing its state; you fix
it by correcting the input before the model's own machinery runs.**

## Next

* If a stronger negative is wanted: 3–5 seeds of the port (cheap, ~12 min each)
  to put a CI on the −0.007 rather than a single point.
* The one remaining "engineering, not structural" escape hatch worth pricing:
  scale train data (revert needs no code — the converter toggle already emits
  train infos; the limit is blob coverage, i.e. the ~400 GB download). State the
  8.9%-data caveat honestly regardless.
* Everything else (per-class honest breakdown, benchmark packaging, paper
  surgery on the retracted vel_warp claim) is unblocked and independent of this.

# Session 2026-08-02 — Stage 08: stop predicting the point, bound the set

## Why a new axis

Every intervention through stage 07 was scored with mAP, which matches a
detection to GT only within 0.5–4 m. That is a point-estimate contract ("the car
is *here*"), and under it the project has five negatives and one input-side
positive. The reframe tested this session drops the point contract:

> Given a detection computed from photons that are k frames old, can we BOUND
> where the object is now, and what does that bound cost?

This is deliberately a third site. The project's law so far is about **state**
(five failures) vs **input** (the honest key-side lift). The envelope is
**output** — post-processing of the emitted detection stream, downstream of every
mechanism stages 04–07 poked at. Note in particular that `vel_warp` is NOT this
experiment: it edited `memory_reference_point` after `post_update_memory`, i.e.
it changed what the NEXT frame reads and never moved the boxes emitted at frame
t. Output-side propagation had never been measured.

## Experiment 1 — the gate: are stale objects MISSED or MISLOCATED?

**Idea / hypothesis.** Inflation can only rescue objects the detector actually
fired on. If staleness makes objects *disappear*, no radius helps and the axis is
dead on arrival. Discriminator: recall at a generous centre distance. If
recall@10m holds while recall@2m collapses, the damage is localisation, not
detection.

Prior evidence pointed both ways: mATE moves only 0.587 → 1.088 m at delay 1
(matched detections are off by ~1 m, well inside the devkit's 2 m and 4 m
thresholds), but motorcycle AP → 0.021 and trailer → 0.000 look like
disappearance. mATE is computed over TPs only, so it is silent about exactly the
objects in question.

**Changes.** [envelope_coverage.py](stages/08_risk_envelopes/envelope_coverage.py)
— no inference. Reuses stage 04's `eval_subset` (cached devkit GT, scene/frame
bookkeeping) and `bootstrap_ci` (cluster bootstrap). GT and predictions are
loaded through the devkit's own `add_center_dist` + `filter_eval_boxes`, so the
object set is exactly the one behind the published mAP numbers. Emission budget
is top-N by score per frame (N=100 primary) rather than a score threshold,
because staleness itself shifts the score distribution.

**Setup.** commit `078e7af` · env `streampetr_env` · CPU only, ~10 min · no W&B
(post-processing of existing artifacts, no training — noting the gap rather than
omitting it). 914 tokens / 23 scenes / 20 800 GT objects after filtering. First k
frames of each scene excluded per config (substitutor clamps to cache length);
fresh is re-evaluated on each config's own token set.

**Metrics.** Recall of GT at increasing centre distance, no propagation:

| config | @0.5 m | @1 m | @2 m | @4 m | @10 m |
|--------|--------|------|------|------|-------|
| fresh | 0.558 | 0.774 | 0.891 | 0.934 | **0.966** |
| d1_naive | 0.164 | 0.348 | 0.582 | 0.798 | 0.951 |
| **d1_key** | 0.200 | 0.424 | 0.671 | 0.860 | **0.957** |
| d2_key | 0.132 | 0.295 | 0.516 | 0.729 | 0.924 |
| d3_key | 0.109 | 0.232 | 0.420 | 0.628 | 0.856 |

@10 m retention: **98.9 % at k=1**, 95.5 % at k=2, 88.4 % at k=3. Per class at
k=1, nine of ten classes retain 96–99 %; only motorcycle (n=95) falls to 81 %.

**Verdict: worked — the gate passes decisively.** mAP says the delay-1 honest-key
system is at 42 % of fresh; existence-recall says 99 %. The 500 ms cliff is
almost entirely a *localisation-precision* phenomenon at the tight thresholds
nuScenes averages over, not a detection failure. Objects are found and put in the
wrong place — which is the precondition the envelope axis needed.

## Experiment 2 — coverage vs cost, and does propagation survive its own null?

**Idea / hypothesis.** Propagate each emitted detection by `v · k · dt`, inflate
to a radius r, and ask what fraction of GT falls inside a same-class envelope.
Three variants so the velocity term is not taken on faith: `static` (no
propagation — the null), `velo` (predicted velocity), `oracle_velo` (velocity of
the nearest same-class GT within 10 m; a diagnostic upper bound, since it reads
GT and may associate the wrong instance for a fast object).

Given Phase 2 (velocity extrapolation contributed 36.15 → 36.14 m on memory
reference points) and `vel_warp` (+0.0015, 13/23 scenes, p=0.68), the prior was
that the velocity term would again be noise.

**Metrics.** Coverage of MOVING objects (≥ 2 m/s; 2 382 of 20 800), `static` →
`velo`:

| config | r=2 m | r=4 m | r=6 m | r=10 m |
|--------|-------|-------|-------|--------|
| fresh | 0.868 | 0.924 | 0.943 | 0.954 |
| d1_naive | 0.295 → 0.397 | 0.534 → 0.671 | 0.714 → 0.878 | 0.905 → 0.940 |
| **d1_key** | **0.282 → 0.616** | **0.586 → 0.854** | 0.796 → 0.920 | 0.935 → 0.943 |
| d2_key | 0.146 → 0.346 | 0.350 → 0.610 | 0.545 → 0.776 | 0.815 → 0.894 |
| d3_key | 0.110 → 0.225 | 0.273 → 0.424 | 0.427 → 0.575 | 0.668 → 0.754 |

Held to the standard that killed `vel_warp` — scene-level paired test, n=22
(one scene has no moving objects in eligible frames), `velo` vs the `static`
null on d1_key:

| r | velo | static | mean Δ | 95 % CI | wins | sign p | significant |
|---|------|--------|--------|---------|------|--------|-------------|
| 2 m | 0.654 | 0.334 | **+0.320** | [+0.231, +0.414] | **20/22** | 0.000 | **YES** |
| 4 m | 0.870 | 0.631 | **+0.239** | [+0.165, +0.319] | **20/22** | 0.000 | **YES** |
| 6 m | 0.930 | 0.827 | +0.102 | [+0.061, +0.146] | 15/22 | 0.134 | CI yes, sign no |
| 10 m | 0.953 | 0.945 | +0.008 | [−0.001, +0.018] | 9/22 | 0.523 | no |

The r=6 row is honest disagreement between the CI and the sign test: the effect
is large where it lands but not present in every scene. At r=10 propagation
correctly does nothing — the static envelope already covers almost everything, so
there is nothing left to win. The pattern is mechanism-consistent: propagation
matters exactly where the bound is tight.

**Oracle velocity buys nothing.** At k=1, `velo` 0.616 vs `oracle_velo` 0.617 at
r=2. At k=3 the predicted velocity is *better* than the oracle (0.225 vs 0.192) —
the head's velocity estimate is fine, and the nearest-GT association degrades
with the gap. **A better velocity head is not the lever.**

**Cost.** Coverage alone is trivially winnable (r → ∞ covers everything), so it is
priced against the fraction of the 50 m ego disc claimed as possibly-occupied by
the union of envelopes (0.073 at r=2, 0.174 at r=4, 0.277 at r=6; essentially
identical for fresh and stale, since both emit ~175 boxes/frame before the
budget). Equivalent radius for fresh-level coverage of moving objects:

| fresh bound | fresh coverage | stale radius needed | radius × | per-envelope area × | claimed-disc area × |
|-------------|----------------|---------------------|----------|---------------------|---------------------|
| 2 m | 0.868 | 4.4 m | 2.2 | 4.8 | **2.6** |
| 4 m | 0.925 | 6.6 m | 1.7 | 2.7 | 1.7 |
| 6 m | 0.944 | 10.2 m | 1.7 | 2.9 | 1.6 |

The two area columns differ because envelopes overlap: πr² grows 4.8× while the
*union* claimed over the disc grows only 2.6×. Quote the union figure — the
per-envelope ratio overstates the cost to a consumer.

**Verdict: worked.** 500 ms of observation debt costs ≈ 2.2× the radius and
≈ 2.6× the claimed area to hold fresh-level coverage of moving objects. That is a
real price but not a prohibitive one, and it is the first positive result in this
project that is about *moving objects* — the population every state-side
intervention failed on.

## The decomposition this makes explicit

`d1_naive` vs `d1_key` on the static null for moving objects is a wash (0.295 vs
0.282), but on all objects it is 0.582 vs 0.671. Add propagation and the moving
numbers separate hard: 0.397 vs 0.616.

| | ego term | object-motion term |
|---|---|---|
| fixed by | honest key-side lift (INPUT) | envelope propagation (OUTPUT) |
| population | all objects, incl. static | moving objects only |
| parameters | 0 | 0 |

The two error sources are **separable, and each has its own zero-parameter fix at
a different site**. Neither substitutes for the other. This is the operational
form of the 2026-07-17 split (ego artefact ≈ 1/6, information loss ≈ 5/6): the
5/6 is not recoverable as a point estimate, but a large part of it is
recoverable as a *set*.

## What this does NOT show

* **Coverage is not precision.** 100 envelopes/frame against ~50 GT objects means
  substantial over-claiming, and the claimed-area fraction is the only thing
  currently pricing it. A planner also needs to know which envelopes are real.
  Nothing here measures false-envelope rate or per-envelope confidence.
* **The radius is swept, not calibrated.** A guarantee ("≥ 90 % coverage") needs
  split-conformal calibration per (class, k) on held-out scenes, not a global
  hyperparameter chosen after seeing the curve.
* **Envelopes are isotropic discs.** Anisotropy (inflate along heading, not
  laterally) is the obvious next win and is untested.
* **No refresh policy exists yet.** "Selective refresh" needs a cost model, and
  the decision granularity is per-frame or per-camera — you cannot recompute one
  object. The deliverable would be a coverage-vs-refresh-rate curve.
* Same 23-scene partial-blob subset as everything else; moving objects are only
  11 % of the GT population here.

## Next

* Anisotropic envelopes + split-conformal calibration of the radius. Both are
  CPU-only on the same saved detections.
* Per-camera refresh policy on top of `AsyncSchedule`, priced against a measured
  cost for one forward pass.
* Prior art to engage before claiming novelty: conformal prediction for object
  detection, probabilistic 3D detection, RSS safety envelopes, forward reachable
  sets. The defensible twist is **envelopes indexed by observation age** plus the
  refresh policy, not inflated boxes per se.

Artifacts: [envelope_coverage.py](stages/08_risk_envelopes/envelope_coverage.py),
[dumps/envelope_coverage.json](stages/08_risk_envelopes/dumps/envelope_coverage.json),
[dumps/envelope_full.log](stages/08_risk_envelopes/dumps/envelope_full.log).

# Session 2026-08-04 — Stage 08 cont.: calibrate the radius, then price the shape

## Why

The swept radius of the previous session is a hyperparameter chosen after seeing
the answer, and it says nothing about a scene the calibration never saw. Split
conformal replaces it with a threshold calibrated on held-out scenes — and in
doing so **changes what the experiment is about**: conformal PINS coverage at
1 − α, so coverage stops being the figure of merit and **claimed area becomes the
free variable** that envelope designs compete on.

**Changes.** [conformal_envelopes.py](stages/08_risk_envelopes/conformal_envelopes.py).
Nonconformity score `s_g = min_i ||x_g − c_i||_(ρ) / σ_i`, so that coverage at
level q is exactly `s_g ≤ q` and one calibration routine serves every design:

* `σ_i = 1 + β·|v_i|·k·dt` — a per-DETECTION scale from its own **predicted**
  speed (never GT, or the method would not be deployable).
* `||·||_(ρ)` — distance in the detection's own propagation frame, stretched by
  an **area-neutral** aspect ratio: semi-axes `(ρqσ, qσ/ρ)` have area `π(qσ)²`
  for every ρ. Verified numerically (unit-ball area = 3.1407 / 3.1412 / 3.1442
  for ρ = 1 / 1.5 / 3 vs π = 3.1416), so any area win is shape, not budget.
* Calibration/test partitions are **scene-disjoint**, drawn once and shared by
  every design so all comparisons are paired.

**Setup.** commit `078e7af` · env `streampetr_env` · CPU, ~12 min · no W&B ·
config `d1_key` (honest key-side lift, k=1) · 20 291 eligible GT objects ·
40 splits of 12 cal / 11 test scenes · α = 0.1 unless stated.

## Result 1 — the guarantee is approximate here, and the shortfall is measured

| β | α | nominal | achieved | sd | splits ≥ nominal |
|---|---|---------|----------|-----|------------------|
| 0 | 0.30 | 0.700 | 0.696 | 0.052 | 0.47 |
| 0 | 0.20 | 0.800 | 0.795 | 0.039 | 0.55 |
| 0 | 0.10 | 0.900 | 0.894 | 0.017 | 0.42 |
| 0 | 0.05 | 0.950 | 0.939 | 0.018 | 0.30 |

Under exchangeability, ~half the splits should reach nominal. We get 0.42 at
α=0.1 and **0.30 at α=0.05**, with the shortfall growing as α tightens. That is
the 2 Hz within-scene correlation (the same objects re-observed for ~20 s)
breaking exchangeability, and it is exactly why the protocol measures achieved
coverage instead of asserting the guarantee. **Quote these bounds as approximate,
with the observed shortfall attached.**

Coverage floor: only **0.82 %** of eligible GT (1.45 % of moving GT) have no
same-class detection anywhere in frame, so 95 % coverage is reachable in
principle — the binding constraint is radius, not detection.

## Result 2 — the per-class radius is the deployable output (and the honest one)

Calibrated radius at 90 % coverage, β=0, ρ=1, median over splits:

| class | q (m) | | class | q (m) |
|-------|------:|-|-------|------:|
| trailer | **inf** | | truck | 10.12 |
| construction_vehicle | **inf** | | barrier | 4.85 |
| motorcycle | **inf** | | **car** | **4.31** |
| bus | 39.45 | | pedestrian | 3.80 |
| bicycle | 32.71 | | traffic_cone | 2.20 |

This is the result to lead with. The envelope is **deployable for the common
classes** — car 4.3 m, pedestrian 3.8 m, cone 2.2 m — and **vacuous for the rest**:
three classes cannot reach 90 % at any radius, and two more demand 30–40 m, which
no planner can use. A single global number would have hidden this completely;
exposing it per class is what the calibration is for.

Independent confirmation worth noting: car's 4.31 m reproduces the 4.4 m
equivalent-radius from the previous session's sweep, by a completely different
route (order statistic on held-out scenes vs interpolation on a pooled curve).

It also explains the aggregate area. Union over ALL classes is 0.467 ± 0.156 of
the 50 m disc; over cars alone it is **0.085 ± 0.009**. The aggregate is set by
whichever tail class drew the largest radius, not by the envelope shape — so the
aggregate column is reported but decides nothing.

## Result 3 — speed-adaptive σ (β) is falsified as a fix

| speed (m/s) | n | β=0 | β=0.5 |
|-------------|---|-----|-------|
| [0.0, 0.5) | 14269 | 0.907 | 0.897 |
| [0.5, 2.0) | 3685 | 0.900 | 0.902 |
| [2.0, 5.0) | 477 | 0.904 | 0.929 |
| [5.0, inf) | 1860 | 0.902 | 0.949 |

The motivation for β was "marginal coverage hides conditional failure". **At β=0
there is no conditional failure to hide** — coverage is ≈0.90 in every speed
stratum. The reason is structural: class-conditional calibration already
conditions on motion implicitly, because class and motion are strongly correlated
in nuScenes (cars and buses move; cones and barriers do not). Raising β
over-covers fast objects, robs static ones (0.907 → 0.897), and costs **1.5× car
area** at equal α. β is retained in the grid as the control that establishes
this, not as a recommendation.

**Verdict: didn't work — and the reason is more useful than the knob would have
been.** Conditioning on class is doing the job people usually reach for a
speed term to do.

## Result 4 — anisotropy helps, slightly, and only when mild

Paired per-split comparison of car claimed area against the disc, β=0, α=0.1.
The ellipse is area-neutral by construction, so any reduction is shape, not
budget. This test was necessary: the across-split spread (±0.009) is larger than
the differences, so the marginal means could not decide it.

| ρ | mean car area | mean Δ | rel | splits better | sign p |
|---|--------------:|-------:|----:|--------------:|-------:|
| 1.00 (disc) | 0.0852 | — | 1.000 | — | — |
| **1.25** | **0.0813** | **−0.0039** | **0.954** | **8/8** | **0.0078** |
| 1.50 | 0.0814 | −0.0039 | 0.955 | 8/8 | 0.0078 |
| 2.00 | 0.0871 | +0.0019 | 1.022 | 2/8 | 0.289 |
| 3.00 | 0.0983 | +0.0131 | 1.153 | 0/8 | 0.0078 |

**Verdict: worked, small.** Mild elongation along the direction of travel buys
**−4.5 % claimed area at identical coverage in 8/8 splits**; strong elongation is
significantly worse. p=0.0078 is the floor for n=8, and the ρ=1.00 row is the
self-comparison (its printed p is an artifact of comparing the baseline to
itself).

The ceiling was predicted in advance and the prediction held: the calibrated
quantile is set by the mostly-static majority of objects, which ρ leaves
untouched by construction (detections below 0.5 m/s stay isotropic). So ~5 % is
about what this knob can be worth as posed. Getting more from shape requires
calibration conditioned on predicted motion — which is exactly what β does, and
β costs more area than it saves. **Both adaptive knobs are bounded by the same
structural fact: this population is mostly static.**

## Methodology bugs found mid-experiment

1. **Area evaluated at a median q̂ while coverage was averaged over splits**
   (real, and it did contaminate output). Per-class order statistics jump between
   splits — the car quantile alone has sd 0.33 m over the range 3.31–4.92 — so
   the first run's area column was jitter (0.36–0.56, non-monotone). Fixed by
   computing area under each split's own calibration; the column is smooth and
   monotone afterwards.
2. **The rng was consumed sequentially inside the design loop**, so designs would
   have received DIFFERENT splits and the pairing the code claimed would not have
   existed. Fixed with `make_splits()`. **This one never contaminated a reported
   number** — it was fixed before the first surviving run. Verified rather than
   assumed: `q_splits` are identical across the v2 and v3 artifacts for every
   design, which could not happen under different partitions given the observed
   per-split spread of q.

Lesson worth carrying: when a comparison is between designs, verify the pairing
is real before reading the numbers — and when claiming a bug invalidated results,
check whether it actually reached them.

## Where stage 08 stands

The deployable statement, with every caveat attached:

> Running one 500 ms-stale camera frame through an honest key-side lift and
> propagating each emitted detection by its own predicted velocity, a
> conformally-calibrated envelope bounds a car's true position within **4.3 m**
> and a pedestrian's within **3.8 m** at 90 % coverage, claiming 8.5 % of the
> 50 m ego disc. Mild along-track elongation removes another 4.5 % of that area.
> Three of ten classes admit no finite bound at this level, and the coverage
> guarantee is approximate on 2 Hz data — 0.894 achieved against 0.900 nominal,
> with only 42 % of held-out splits reaching it.

## Open

* Deeper staleness (k=2, 3) under the same calibration — `--config d2_key`.
  Not yet measured. Expect more classes to go infinite than at k=1, since the
  gate table already shows recall@10m falling 0.957 → 0.924 → 0.856 across
  k=1/2/3, but the floor itself (unreachable fraction at ANY radius) has only
  been computed for k=1.
* The three infinite classes need their own treatment: either a larger
  calibration set (they are rare on 23 scenes — trailer has 26 objects) or an
  explicit "no bound available" output, which is itself a safe and honest signal
  to a planner and arguably the right answer.
* Still nothing measures PRECISION. ~100 envelopes/frame against ~50 GT objects
  means substantial over-claiming, and claimed area is the only proxy. A planner
  needs to know which envelopes are real, not just that the true object is inside
  one of them.
* Per-camera refresh policy remains the unbuilt piece that would turn this into
  a compute decision rather than a bound.

Artifacts: [conformal_envelopes.py](stages/08_risk_envelopes/conformal_envelopes.py),
[dumps/conformal_d1_key_v3.json](stages/08_risk_envelopes/dumps/conformal_d1_key_v3.json),
[dumps/conformal_d1_key_v3.log](stages/08_risk_envelopes/dumps/conformal_d1_key_v3.log)
(v3 = v2 plus the per-split area vectors needed for the paired ρ test; the two
agree exactly on every shared quantity).
# Session 2026-08-04 (cont.) — Stage 08: precision, or what the other envelopes are doing

## Why

Coverage says the true object is inside SOME envelope. It says nothing about the
other ~100 envelopes per frame, and a consumer handed 100 "might be here" regions
that are mostly about nothing has not been helped. Every stage-08 number so far
measured only the first half.

**Changes.** [envelope_precision.py](stages/08_risk_envelopes/envelope_precision.py).
The insight that shapes the experiment: after conformal calibration the radius is
no longer free, so the remaining knob is the **emission budget**, and it cuts
both ways — fewer envelopes means fewer false ones but a larger calibrated radius
to still reach 90 % coverage. So the budget is swept, the radius RE-CALIBRATED at
each budget, and the designs compared at equal coverage.

**Setup.** commit `078e7af` · env `streampetr_env` · CPU, ~15 min · no W&B ·
α=0.1, β=0, ρ=1.25 · 40 splits (4 for the area raster) · `--classes car`.

## Result — the cost of lateness confirmed a third time, and a misattribution corrected

Car only, coverage pinned at ~0.89, stale (`d1_key`) vs fresh:

| budget | stale r | stale area | stale prec | fresh r | fresh area | fresh prec |
|-------:|--------:|-----------:|-----------:|--------:|-----------:|-----------:|
| 10 | 13.35 m | 0.2279 | 0.776 | 12.99 m | 0.2216 | 0.754 |
| 30 | 4.64 m | 0.0724 | 0.435 | 2.53 m | 0.0349 | 0.360 |
| **50** | **3.90 m** | **0.0714** | **0.327** | **1.86 m** | **0.0302** | **0.255** |
| 100 | 3.50 m | 0.0739 | 0.258 | 1.58 m | 0.0302 | 0.190 |
| 300 | 3.43 m | 0.0735 | 0.246 | 1.56 m | 0.0310 | 0.181 |

1. **Cost of 500 ms at equal coverage: 2.1× radius, 2.36× claimed area.**
   Independently reproduces the 2.2× / 2.6× from the sweep and the 4.31 m
   conformal radius, by a third route. This number is now well established.

2. **Low precision is NOT caused by staleness — this corrects the previous
   session's reading.** The fresh detector scores 0.255 at the same operating
   point, i.e. three quarters of ITS envelopes are also empty. The earlier
   "57 % of envelopes are about nothing" was factually right and causally wrong:
   it is the detector's own false-positive rate at ~50 boxes/frame, present with
   or without delay. Staleness costs SPACE, not precision.

3. **Optimal budget is 30-50 for both.** Below it (10) precision looks great
   (0.78) but the radius explodes to 13 m and area triples; above it area is flat
   while precision keeps falling. This is the operating point the refresh policy
   should assume.

Redundancy (hitting envelopes per covered object) is 2.0-2.2 stale vs 1.5-1.6
fresh: the late system needs more envelopes to cover the same car.

## Two metric defects found, one fatal

1. **`wasted` as defined is not waste.** "Claimed area further than 2 m from an
   annotated object" penalises a bound for being a bound: with a 4 m calibrated
   radius, a PERFECT envelope is already ~77 % "wasted", and the metric saturates
   at 0.94-0.96 for every budget in the all-class run. Retained in the output as
   a crude free-space indicator only, with this caveat attached; **no conclusion
   rests on it.** It becomes meaningful only where the radius is below the 2 m
   core (fresh at budget >= 50, where 0.75 genuinely reflects false envelopes).
   The honest cost reference is the fresh detector's own claimed area, which is
   what the table above uses.

2. **Precision-by-containment rewards a larger radius.** Stale "beats" fresh on
   precision (0.327 vs 0.255) purely because its envelopes are bigger, not
   because it is more discriminative. **Precision must not be compared across
   configurations with different radii** — only within one, across budgets.

Aggregate (all 10 classes) numbers are reported in
[dumps/precision_d1_key.json](stages/08_risk_envelopes/dumps/precision_d1_key.json)
but remain dominated by bus (~39 m) and bicycle (~33 m) radii, as in the
conformal session. `--classes car` is the interpretable view.

## What stage 08 now supports, end to end

> Under a 500 ms camera delay, correcting the input geometry and propagating each
> emitted detection by its own predicted velocity yields conformally-calibrated
> envelopes that contain the true car in 89 % of held-out cases within **3.9 m**,
> at **2.4× the space** a fresh detector needs for the same guarantee. The
> envelope set's precision (0.33) is inherited from the detector, not from the
> delay.

## Next

* **Refresh policy** — now unblocked, since it needs the operating point
  (budget 30-50) this session established. Deliverable: coverage held vs fraction
  of frames/cameras refreshed, which needs a measured cost for one forward pass.
* Deep staleness k=2,3 and the three infinite classes: still open, still unmeasured.
* Precision could be made radius-independent (e.g. envelopes per unit area
  containing an object) if a cross-config comparison is ever needed.

# Session 2026-08-04 (cont.) — Stage 08: the refresh policy

## Why

The envelope bounds where an object is given that the frame is k steps old. But
k is not fate — a stack can spend a forward pass and reset it to 0. This is the
systems half of the contribution, and the precision session unblocked it by
fixing the operating point (budget 50).

**Changes.** [refresh_policy.py](stages/08_risk_envelopes/refresh_policy.py).
Detections already exist for k = 0,1,2,3, so a policy is just an assignment of a
staleness to every frame and the envelope set is read off the matching config —
no inference. Radius is calibrated **per (k, ego displacement bucket)**
(Mondrian conformal), so coverage stays pinned and the comparison is on space
and compute. Policies: `periodic(N)`, `ego(d)` (odometry only), `obj(d)`
(fastest predicted object speed among the detections already emitted), and
`random(p)` as the control.

**Setup.** commit `078e7af` · env `streampetr_env` · CPU, ~25 min · no W&B ·
cars only · budget 50 · α=0.1 · 20 splits (2 for area).

## Result 1 — the compute/bound curve

| refresh rate | mean radius | claimed area |
|-------------:|------------:|-------------:|
| 1.00 | 1.97 m | 0.0235 |
| 0.50 | 2.85 m | 0.0476 |
| 0.35 | 3.97 m | 0.0737 |

**Halving the forward passes roughly doubles the claimed area** at the same 90 %
coverage. That is the price of compute, stated in the only currency the envelope
framing has.

## Result 2 — how you spread refreshes matters; which trigger you use does not

Claimed area at MATCHED refresh rate (linear interpolation within each family):

| rate | periodic | ego | obj | random |
|-----:|---------:|----:|----:|-------:|
| 0.45 | 0.0564 | **0.0549** | 0.0583 | 0.0633 |
| 0.50 | **0.0482** | 0.0508 | 0.0533 | 0.0575 |
| 0.60 | **0.0429** | **0.0429** | 0.0434 | 0.0484 |
| 0.70 | 0.0381 | **0.0349** | 0.0364 | 0.0422 |
| 0.80 | 0.0332 | **0.0288** | 0.0315 | 0.0360 |

* **`random` is worst at every matched rate.** Robust and the one clear ordering:
  spending the same compute blindly costs 10-20 % more space. Scheduling matters.
* **No policy clearly beats `periodic`.** `ego` wins at high rates (~10 % at
  0.8), `periodic` wins mid-range (~5 % at 0.5), `obj` sits between. Differences
  are 5-13 % with **no confidence intervals** (2 area splits), so none of this is
  established. An earlier reading of a single rate (0.5) suggested periodic beat
  both adaptive triggers; that does not survive looking at the whole range and is
  retracted.

## Result 3 — ego displacement is the WRONG trigger now, and that follows from our own result

Calibrated radius (m) per (k, ego displacement since capture):

| k | <0.5 m | 0.5-2 m | 2-4 m | >4 m |
|---|-------:|--------:|------:|-----:|
| 0 | 1.94 | — | — | — |
| 1 | **3.48** | **2.51** | 3.32 | 4.62 |
| 2 | 5.08 | 3.95 | 4.45 | 7.29 |
| 3 | 6.32 | 5.80 | 5.96 | 12.28 |

**Non-monotone**: at k=1 a nearly-stationary ego needs a LARGER radius (3.48 m)
than one that moved 0.5-2 m (2.51 m). The 2026-07-17 session found damage scaling
with ego displacement — but that was the NAIVE harness. The honest key-side lift
removed the ego term (measured then as "inert at zero ego motion, helping in
proportion to displacement"), so on the corrected system the residual is object
motion, and odometry does not predict it. Ego displacement stratified the error
of the harness we fixed, not of the one we now run.

That is why `obj` was added mid-session as the trigger the corrected hypothesis
implies. It does not rescue the story either — it tracks `ego` closely and beats
only `random`.

Structural reading of why adaptivity pays so little: the radius is convex in k
(1.94 → 3.5 → 4.5 → 6), so for a fixed refresh budget, spreading refreshes evenly
minimises the bound, and a trigger only wins if per-frame damage varies more than
the staleness effect itself. After the input fix, it does not.

## Limits (both bind hard)

1. **k ≤ 3 clamp.** No detections exist beyond 3 frames, so every policy is
   forced to refresh at age 3 and none can go below a 0.346 refresh rate. The
   low-compute regime — the one a scheduler would actually be interesting in —
   is inaccessible with these dumps. All families converge there by construction.
2. **Trajectory mismatch.** The k=2 detections come from a run where EVERY frame
   used 2-frame-old features, not from one that refreshed and then coasted. The
   photon age is right; the memory history is not, since StreamPETR carries
   `memory_embedding` forward. Same defect that produced the Phase 4 chimera.
   These are first-order cost estimates, **not a simulation of a running policy**.
   Closing it needs the policy in the inference loop, which is GPU work.

## Next

* Dump k=4..8 to open the low-refresh-rate regime, then re-run — this is the
  cheapest way to make the policy result interesting rather than clamped.
* Put a CI on the policy differences (more area splits) before any of the 5-13 %
  gaps is quoted.
* A policy-in-the-loop GPU run is what would turn all of this from an estimate
  into a measurement.

Artifacts: [refresh_policy.py](stages/08_risk_envelopes/refresh_policy.py),
[dumps/refresh_policy.json](stages/08_risk_envelopes/dumps/refresh_policy.json),
[dumps/refresh_policy.log](stages/08_risk_envelopes/dumps/refresh_policy.log).

# Audit: re-checking every claim from stage 04 onward

> **Read this before citing any number from stages 04, 07 or 08.** A full
> read-only review of the code behind every claim, plus re-evaluation of the
> saved `results_nusc.json` / conformal artifacts on CPU. Earlier sections are
> left intact per convention; this section says which of their conclusions
> change. Headline: one implementation bug that reopens a "negative" result,
> three claims that the artifacts contradict, and a set of statistics that
> were over-read. The honest key-side lift at k=1 and the stage-08
> cost-of-staleness numbers survive.

## Bug: `vel_warp` added a lidar-frame velocity to a global-frame point

**What.** `MemoryMitigator` runs after `simple_test`, i.e. after
`post_update_memory`. By then `memory_reference_point` has been moved to the
GLOBAL frame ([streampetr_head.py:372](StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L372)),
but `memory_velo` is the raw regression output, which lives in lidar(t) — the
converter rotates GT velocity global → lidar
([nuscenes_converter.py:270-275](StreamPETR/tools/data_converter/nuscenes_converter.py#L270)).
`vel_warp` / `vel_warp_gated` added one to the other. The lidar→global yaw has
median ~55° over the 914 val frames (>90° in 29%), so the push pointed in an
essentially arbitrary direction. Units are fine (both metres).

**Fix.** `velo_lidar_to_global()` in
[measure_stale_map_mitigation.py](stages/04_streampetr_latent/measure_stale_map_mitigation.py)
rotates by `data['ego_pose'][:3,:3]` (lidar→global) before the push. Validated
on CPU against the devkit: 137 moving GT objects (>1 m/s) from the val pkl,
rotated `gt_velocity` vs `nusc.box_velocity` → max abs error **0.046 m/s**; the
unrotated (old) velocity is off by **4.24 m/s** on average.

**LTC has the same flaw** in its analytical base
([ltc.py](stages/04_streampetr_latent/ltc.py), comment added). Its learned
delta was trained on top of the wrong push, so it is not drop-in fixable —
re-dump + retrain. Not done; the LTC line stays closed.

**Re-run prepared, NOT yet submitted:**
[submit_velwarp_fixframe.sh](stages/04_streampetr_latent/submit_velwarp_fixframe.sh),
4 tasks: honest-key vel_warp at k=1/2/3 (compare 0.1849 / 0.1007 / 0.0731) and
naive vel_warp at k=1 (compare 0.1331 — the paper's original condition). New
dump dirs `*_fixframe`; old dumps untouched.

**What this reopened.** "vel_warp is noise (13/23, p=0.68)" and the vel_warp
row of the unified table had tested a broken implementation, so the
state-vs-output law was in part buggy-vs-correct until re-run.

### Re-run result: fixing the frame changes nothing — vel_warp is still noise

Job 52673354 (4 tasks, ~4 min each, one A100). Pooled mAP and scene-level
paired tests (n=23, `dumps/bootstrap_velwarp_fixframe_*.json`), old buggy
version alongside:

| condition | baseline | buggy pooled | fixed pooled | fixed per-scene Δ [95% CI] | wins | sign p |
|-----------|---------:|-------------:|-------------:|---------------------------|------|--------|
| key k=1 | 0.1849 | 0.1906 | 0.1880 | +0.0002 [−0.0015, +0.0020] | 11/23 | 1.00 |
| key k=2 | 0.1007 | — | 0.1022 | +0.0006 [−0.0005, +0.0018] | 12/23 | 1.00 |
| key k=3 | 0.0731 | — | 0.0733 | −0.0002 [−0.0013, +0.0008] | 11/23 | 1.00 |
| naive k=1 | 0.1331 | 0.1378 | 0.1375 | +0.0017 [−0.0002, +0.0040] | 14/23 | 0.41 |

(Buggy per-scene Δ for comparison: −0.0004 / −0.0003 / −0.0007 / +0.0015.)

**Verdict: didn't work — and the bug was real but irrelevant.** Correct and
arbitrary-direction pushes of the new block's reference points give the same
mAP to within ±0.001. The prior verdict stands for a better reason than it was
reached: **the detector is insensitive to where the 256 propagated new-block
reference points sit**, so no velocity projection of them — right or wrong —
can move mAP. This restores vel_warp as a support for "state edits fail", and
sharpens it: the contrast with stage 08 is not buggy-vs-correct, since
output-side propagation of the *emitted* boxes (+0.32 coverage at 2 m) works
while the same velocity applied to *memory* is inert. The paper's original
"+3–5%" (naive, 0.1378) is also noise with the fix (14/23, p=0.41).

## Stage 04: claims the artifacts contradict

**1. The key lift helps LESS with depth, not more.** The k=2/3 key-only runs
(listed as "in flight" in the harness-audit section) exist:

| k | naive | honest key | Δ pooled | per-scene Δ [95% CI] | wins | sign p |
|---|-------|-----------|----------|----------------------|------|--------|
| 1 | 0.1331 | 0.1849 | +0.0518 | +0.0348 [+0.024, +0.047] | 23/23 | 0.000 |
| 2 | 0.0774 | 0.1007 | +0.0233 | +0.0153 [+0.006, +0.027] | 19/23 | 0.003 |
| 3 | 0.0610 | 0.0731 | +0.0121 | +0.0059 [−0.003, +0.017] | 13/23 | 0.68 (n.s.) |

Share of the gap-to-fresh closed (stale-only tokens): **16.7% → 6.2% → 2.9%**.
The advance prediction ("ego displacement grows ~3 m/frame while the fix stays
exact, so it should help more") failed. Something other than lift geometry
dominates at depth.

**2. "Ego artefact ≈ 1/6, object motion ≈ 5/6" is not supported.** Under the
honest key lift, *stationary* classes still collapse with ego displacement
(`stratified_egomotion_d1_fixed.json`, key/fresh retention by stratum):

| ego displacement | traffic_cone | barrier |
|------------------|-------------:|--------:|
| [0, 0.5) m   | 98.0% | 89.6% |
| [0.5, 2) m   | 67.3% | 69.6% |
| [2, 4) m     | 54.8% | 53.2% |
| [4, ∞) m     | 41.4% | 44.9% |

Cones cannot move, so the residual after an EXACT geometric correction is still
ego-driven — viewpoint/FOV/occlusion change, the 2D ROI top-k choosing query
spawn locations from stale pixels, or new-vs-carried query inconsistency. The
1/6 is the part the lift fixes; the remaining 5/6 is NOT shown to be object
motion. Also, the gain is not "in proportion to ego displacement":
+24.0 / +9.6 / +12.1 pp is non-monotone.

**3. The mechanism check for the lift has no artifact.** The 3.335 m /
0.000000 m / 9.5e-13 figures are not produced by any script in the repo, and
the `honest_staleness.py` docstring says 3.09 m. "Honest error = 0" is
tautological if it used the same composition. The lift maths itself was
re-derived and is correct (`ego_pose` = lidar→global, composition direction
right, image-aug matrices on the left of `lidar2img`, no leak into buffers).

## Stage 07: the age-port test started below its baseline

`install_age_port` always feeds the TRUE `(T, dt)`, and at init the port is an
exact copy of the shared MLN / time embedding. So at k=1 the untrained port
reproduces honest **query/both** conditioning (0.1242, known harmful), not
honest-key (0.1849). "Bit-identical at dt=0" is true but irrelevant at eval,
where almost every frame has dt>0; "can only move where delay>0" is false after
training (k=0 frames also flow through the port).

Read correctly: training moved it ~0.124 → 0.171, i.e. it *recovered most of
the query-side damage*; it did not show that a trained port cannot help. The
unreported 40-step smoke run scored **0.1801** — longer training did worse.
Other open confounds: the 62 train scenes are in-sample for the frozen
detector; training delay is uniform 0–3 while eval is k=1; one seed, no LR
schedule; epoch-mean loss barely moved (10.75 → 10.53).

Also: by the paired tests (sign p=0.011, Wilcoxon p=0.015) the port is
significantly WORSE than honest-key; "not significant" came only from the
bootstrap CI touching +0.0001. Missing control: eval at 0 training steps.
**"Given every advantage" and "the chimera does not dissolve" are not earned.**

## Statistics conventions that bias numbers (no sign flips)

* **Unified intervention table mixes units.** Key row is a pooled Δ vs naive
  (+0.052); every other row is a mean per-scene Δ vs honest-key. Pooled
  equivalents: age port −0.0136, (buggy) vel_warp +0.0057.
* **Per-scene mAP counts classes absent from a scene as AP=0** (devkit
  `no_predictions()` when npos=0, mean still over 10 classes). Same for every
  config, so signs/wins hold, but per-scene deltas are attenuated.
* **Sign test counts ties as losses** (one tie, d3 key).
* **Headline mAPs include the first-k fresh-fallback frames.** Stale-only
  tokens: naive k=1 0.1312, key k=1 0.1841, key k=2 0.0970, naive k=3 0.0551
  (headline inflated ~11%), key k=3 0.0665, age port 0.1702. Every k=1
  conclusion survives (key Δ +0.0529, 16.7% of gap).

## Stage 08: corrections

**Survive and reproduce from artifacts:** gate table and @10 m retention;
moving-object coverage and the velo-vs-static paired test (20/22); oracle
velocity buys nothing; car q=4.31 m (sd 0.33), pedestrian 3.80 m; cost of
500 ms ≈ 2.1× radius / 2.4× area; precision rows; refresh compute/bound table.
Frames are consistent (devkit boxes and velocities both global, ego disc on
`ego2global_translation`); no GT leakage in any deployable path; using true
keyframe gaps instead of dt=0.5 changes car q by 0.004 m.

**Wrong or over-read:**

1. **"Three classes admit no finite bound" is wrong for trailer.** Per-split q
   (v3, β=0, ρ=1, α=0.1): trailer is inf in 20/40 splits (7 with no calibration
   trailers, 13 with too few, n<9) and **6.14 m in every finite split**; only
   3.8% of trailers are unreachable. Bus 39.45 m is the median of 21 finite +
   19 inf; bicycle has 10/40 inf. Construction vehicle (28/40 inf) and
   motorcycle (24/40) are genuine "no bound at 90%". `np.nanmedian` does not
   skip inf — report the count of inf splits, not a median.
2. **Rare-class finite radii do not hold on held-out scenes:** held-out
   coverage where q is finite — bus 0.72, construction_vehicle 0.64, motorcycle
   0.61, trailer 0.88. Test objects in q=inf classes count as covered but cost
   zero area (`claimed_area` skips non-finite r). Counting them uncovered,
   achieved coverage is **0.866**, not 0.894; the 0.467 all-class area is an
   underestimate.
3. **The α table was copied from the superseded v1 run**, not v3. v3: α=0.2
   0.797/0.040/0.50; α=0.1 0.894/0.022/**0.40** (not 0.42); α=0.05
   0.942/0.016/0.35. So "the rng bug never contaminated a reported number" is
   false for this table; the qualitative reading is unchanged.
4. **ρ=1.25 "8/8, p=0.0078" is pseudo-replication:** overlapping resamples of
   the same 23 scenes, area rasterised over all frames on the same detections,
   only q varies. Direction (−4.5%) probably real; "significant" is not.
5. **"Independent confirmation by a third route" compares different
   quantities:** 4.4 m = moving objects, all classes; 4.31 m = cars, all
   speeds, all-class top-100; 3.90 m = car-only top-50, ρ=1.25. Note
   `--classes car` filters BEFORE the top-N cut, so "budget 50" there means
   top-50 *car* detections (~39/frame); all-class budget 50 gives q_car 7.21 m.
6. **Speed-conditional table is in-sample** (median q applied to calibration
   scenes too). Held-out: 0.893 / 0.892 / 0.904 / 0.888. Conclusion (no
   conditional failure) survives.
7. **Refresh areas rest on 2 splits, the two lowest-q ones.** Fresh car area
   per split ranges 0.020–0.043 (2×), dwarfing the 5–13% policy gaps; rate-1.0
   area 0.0235 disagrees with the precision session's 0.0302 for the same
   operating point. The `random` control is one realisation (and clamping moves
   its rates: 0.5→0.559). Matched-rate rows 0.6–0.8 for periodic/random are
   linear chords with no data behind them. **None of the policy ordering is
   established**, including "random is worst".
8. **"Optimal budget 30–50" does not hold:** fresh ties at 50 and 100; stale is
   flat (within 3.5%) from 30 to 300.
9. Small code defects: a (k, bucket) group with no frame in the area sample
   returns area 0 but keeps full weight (`refresh_policy.py:331-337`, favours
   random); the first-k fresh-fallback exclusion is missing from the area
   rasters (coverage, conformal), detection precision, and refresh calibration.
   Coast frames are not free either: d_k dumps still run the heads.

**Superseded stage-08 dumps:** `conformal_d1_key.{json,log}` (v1),
`conformal_d1_key_v2.*` (= v3 minus per-split areas), `full_run.log`, all
`_smoke*` / `_prec_smoke` / `_refresh_smoke*`. Current: `envelope_coverage.json`,
`conformal_d1_key_v3.*`, `precision_*`, `refresh_policy.*`.

## What the project can claim right now

* **Solid:** honest key-side lift at k=1 (+0.052 pooled / +0.035 per-scene,
  23/23, inert at zero ego motion). Stage-08 cost of 500 ms staleness (≈2.1×
  radius, ≈2.4× area at equal coverage), output-side velocity propagation
  (+0.32 coverage at 2 m on moving objects, 20/22 scenes), car/pedestrian
  90% envelopes ≈4 m. Suppression (zero/freeze) hurts.
* **Re-confirmed with the frame fix:** vel_warp is noise at k=1/2/3 under both
  harnesses; memory reference-point edits are inert.
* **Needs restating:** the 1/6 vs 5/6 split; key lift at depth; the age-port conclusion; rare-class
  "no bound"; refresh-policy ordering; everything the paper says (below).
* **paper.tex is stale throughout:** sells the retracted vel_warp as the
  positive result, quotes the naive-harness 70% collapse, contains numbers with
  no source in the repo (BEVFusion 1.70–1.78× speedups — stage 03 says that
  checkpoint was broken; 4.59× scaling timings vs ~7× in stage-05 notes;
  hardware/power/EV-range costs), misdescribes MamBEV's Mamba as temporal, and
  omits the key lift and stage 08 entirely. `jiang2020clockwork` has the wrong
  authors. Needs a reframe, not edits.

## Repo hygiene found in the same pass

* **The code is not in git.** Commit `078e7af` holds 5 files; every "Setup:
  commit 078e7af" line above points at a commit that does not contain the code
  that produced the result. ~486 MB of unreachable blobs sit in `.git`.
* Deletion candidates (~74 GB) were listed separately; the ones worth keeping
  are all `results_nusc.json`, `sweeps/CAM_*` (12 Hz — enables sub-500 ms
  delays), `checkpoints/mambev/`, `data_full/nuscenes_infos_val.pkl`.

## Next

* ~~Submit `submit_velwarp_fixframe.sh`~~ DONE — fixed vel_warp is noise (see
  re-run result above). Memory ref-point edits are inert, so velocity
  correction belongs on the emitted boxes (stage 08), not in memory.
* Age port: eval at 0 steps (the missing control), then 3 seeds, k=1-weighted
  training delays.
* Stage 08: apply the code fixes above (area for inf classes, zero-area groups,
  first-k exclusion everywhere), pool rare classes into super-classes for
  calibration, scene-level conformal, ≥20 area splits and averaged `random`
  before quoting any policy gap.
* Zero-GPU coasting baseline (propagate last fresh boxes by v·age) — the
  baseline every stale-inference config must beat.
* Sub-500 ms delays from the 12 Hz camera sweeps.

# Stage 09: realistic delays (80–500 ms) and per-camera asynchrony

> Merged from two write-ups of the same experiment (two sessions recorded it);
> numbers agree. Stale-only tokens are the primary table; pooled all-token mAPs
> are quoted where a comparison uses them.

## Why

Every staleness number before this used whole 500 ms keyframe steps — an extreme
delay. Real stacks live at 50–250 ms, and the "coordinate lie" should already
matter there: at 15 m/s the ego moves 1.5 m in 100 ms, the same scale as the
0.5–4 m mAP thresholds. Second question: the backbone runs per camera, so a
stack can refresh only some cameras per tick. Does the *pattern* of camera ages
matter at equal compute, or only the mean age?

## Method — data-side only, no model hooks

nuScenes cameras record 12 Hz sweeps between the 2 Hz keyframes, so for every
keyframe and camera we step back j sweeps (~80 ms each) along the sample_data
`prev` chain.

* [build_sweep_index.py](stages/09_subframe_async/build_sweep_index.py): per
  (sample, camera, j=0..6) the sweep's path, capture time, and HONEST extrinsics
  = the converter's own `obtain_sensor2top` on the sweep (camera pose at its
  capture time, in the keyframe's lidar frame) — the honest key-side lift, exact
  per camera. **Gate:** j=0 reproduces the pkl to 4.6e-13 m and every path, 0 of
  ~37k sweep images missing. Realised mean ages j=1..6: 78 / 168 / 255 / 333 /
  423 / 509 ms on stale-only tokens (76 / 164 / 249 / 325 / – / 496 ms over all
  tokens; j=6 lands on the previous keyframe 75% of the time). Per-(sample,
  camera) ages are saved in each dump's `realised_ages.json`.
* [eval_async_cams.py](stages/09_subframe_async/eval_async_cams.py): rewrites
  `data_infos` camera entries in memory, then the stock test loop. `naive` =
  sweep image + keyframe extrinsics; `honest` = sweep image + its true
  extrinsics. The whole backbone runs on the real older pixels and memory evolves
  as a real async stack would feed it. Only the key side is corrected; the
  query-side timestamp is untouched.
* Schedules: `uniform:J`; `rr:P` = 6/P cameras refreshed per tick, round-robin
  (ages 0..P−1 coexist); `whole:P` = all 6 refreshed every P ticks. rr:P and
  whole:P have equal compute and equal mean age. (`rand:J` was added later for
  stage 10d.)
* [analyze_async.py](stages/09_subframe_async/analyze_async.py): stage-04
  SubsetEvaluator + paired scene tests; stale-only tokens drop the 23 scene
  starts where no older sweep exists; fresh re-evaluated on the same tokens.

Side finding: the mmdet3d converter already gives each keyframe camera its OWN
capture-time ego pose, so the 42.6 ms intra-frame camera spread is compensated
in pose (not in timestamp). The earlier claim that "every method collapses six
cameras to one pose" is wrong for pose in mmdet3d-converted infos.

**Setup.** job 52739475, 17 tasks, ~5 min each on one A100 · config
`stream_petr_r50_flash_704_bs2_seq_90e` · no W&B (eval only).

**Gates.** `uniform:0` data is bit-identical to fresh and gives **0.4448**
exactly. Cross-check with the old feature-substitution harness at ~500 ms: naive
0.1301 vs 0.1331, honest 0.1807 vs 0.1849 — two independent implementations
agree within 0.004.

## Result 1 — even 80 ms costs real mAP, and the honest lift matters most there

Stale-only tokens (n=891), fresh on the same tokens = 0.4488 (pooled all-token
mAP in brackets, fresh 0.4448):

| mean age | naive mAP | honest mAP | naive loss | honest loss | gap closed | per-scene Δ [95% CI] | wins |
|---------:|----------:|-----------:|-----------:|------------:|-----------:|----------------------|------|
| 78 ms  | 0.3828 [0.3800] | 0.4077 [0.4041] | 14.7% | **9.1%**  | **37.8%** | +0.0168 [+0.010, +0.024] | 19/23 |
| 168 ms | 0.2879 [0.2871] | 0.3304 [0.3290] | 35.8% | 26.4% | 26.4% | +0.0324 [+0.023, +0.042] | 20/23 |
| 255 ms | 0.2241 [0.2244] | 0.2733 [0.2729] | 50.1% | 39.1% | 21.9% | +0.0371 [+0.027, +0.048] | 21/23 |
| 333 ms | 0.1879 [0.1888] | 0.2373 [0.2372] | 58.1% | 47.1% | 18.9% | +0.0364 [+0.026, +0.047] | 21/23 |
| 509 ms | 0.1282 [0.1301] | 0.1799 [0.1807] | 71.4% | 59.9% | 16.1% | +0.0344 [+0.023, +0.047] | 21/23 |

vs fresh at ~78 ms (paired, all tokens): naive −0.039 [−0.050, −0.028], 1/23
scenes; honest −0.022 [−0.029, −0.015], 1/23.

**Verdict: worked — both predictions confirmed.**
1. Staleness is not an exotic 500 ms effect: **one ~80 ms-old frame on all
   cameras already costs 14.7% mAP** under the synchronous interface.
2. The honest lift closes a **larger share of the gap at shorter delays**
   (37.8% at 78 ms → 16.1% at 509 ms), significant at every delay. Its absolute
   gain plateaus at ~+0.035 per scene from ~250 ms on while the gap keeps
   growing — a fixed ego-geometry component plus a growing one the lift cannot
   touch.

Still unexplained: with honest geometry, 78 ms costs 9.1%. Object motion in
78 ms is < 1 m for almost everything, so the residual probably includes the
fresh-carried-memory vs stale-new-query inconsistency and the unconditioned
timestamp, not just object motion.

## Result 2 — at equal compute, only the mean age matters

All tokens, per-scene paired tests:

| schedule (honest) | refresh | mean age | mAP | paired vs whole-frame |
|-------------------|---------|---------:|----:|-----------------------|
| every camera every tick | 6 cams/tick | 0 | 0.4448 | — |
| `rr:2` round-robin | 3 cams/tick | 38 ms | 0.4246 | +0.002 [−0.001, +0.004], 15/23 (n.s.) |
| `whole:2` | all 6 every 2nd tick | 39 ms | 0.4241 | — |
| `rr:3` round-robin | 2 cams/tick | 80 ms | 0.3877 | −0.002 [−0.007, +0.003], 11/23 (n.s.) |
| `whole:3` | all 6 every 3rd tick | 80 ms | 0.3883 | — |
| `rr:2` naive | 3 cams/tick | 38 ms | 0.4108 | |
| `rr:3` naive | 2 cams/tick | 80 ms | 0.3440 | |

Honest vs naive under rr: +0.0137 (19/23) at ½ compute and +0.0230 (19/23) at
⅓, both significant.

**Verdict: mixed.** Round-robin vs whole-frame is a wash — mixing fresh and
stale cameras in one frame costs nothing extra (no cross-camera inconsistency
penalty), but gains nothing either. The mean age prices a compute budget:
**half the camera-backbone compute costs 4.5% mAP with honest geometry**
(0.4448 → 0.4246; paired −0.009 [−0.015, −0.004], 7/23) vs 7.6% naive; a third
costs 12.8% honest vs 22.7% naive. Honest geometry roughly halves the price of
skipping camera passes. Damage looks roughly additive per camera: half the
cameras at ~80 ms costs about half of all six at ~80 ms (4.5% vs 9.1%).

## Caveats

* The model runs only at keyframes (2 Hz), so memory updates less often than in
  a real 12 Hz stack. Ages are right; the memory history under rr/whole is an
  approximation that hurts them more than the uniform-delay arm. The
  equal-compute comparison is fair between the two patterns, not a full 12 Hz
  simulation.
* Pooled mAPs include the ~3% of (sample, camera) pairs at scene start that clamp
  to fresh (same for all configs); stale-only mAP for rr/whole is not comparable
  across the two (different token subsets), so the paired all-token tests are
  the comparison there.
* Same 23 scenes / one model / one seed as everything else.

## Next (status)

* ~~Delay augmentation during fine-tuning~~ → stage 10.
* Isolate the 9% residual at 78 ms: query-side timestamp only; per-class
  breakdown (static classes should be ~free at 78 ms if geometry were the only
  issue). Not done.
* Stage-08 envelopes at 76–250 ms (radius vs age at 80 ms steps). Not done.
* Smarter-than-round-robin refresh (e.g. cameras facing the direction of
  travel); Result 2 suggests patterns matter little unless they lower mean age
  where it counts. Not done.

Artifacts: `stages/09_subframe_async/dumps/async_*` (results + realised ages),
`dumps/analysis_async.json`, `dumps/bootstrap_*_honest_vs_naive.json`,
`dumps/bootstrap_rr*_vs_whole*.json`, `dumps/bootstrap_vs_fresh.json`,
`dumps/sweep_index_val.pkl`,
[submit_async_cams.sh](stages/09_subframe_async/submit_async_cams.sh).


# Stage 10: fine-tuning StreamPETR for sub-keyframe delay

## Why

Stage 09 priced realistic delay (76 ms already costs 14.6% mAP) and showed the
honest lift closes 37% of it at 76 ms. Every earlier "train for it" attempt was a
small bolted-on module on a frozen model (LTC, age port). This is the untried
arm: fine-tune the model's own neck + heads with random per-camera delay and
honest geometry, with a control arm that is identical except for the delay.

## What was built

* [build_sweep_labels2d.py](stages/10_delay_finetune/build_sweep_labels2d.py):
  2D ROI labels for sweep images — the converter's `get_2d_boxes` logic with
  devkit boxes interpolated to the sweep's capture time (objects as they appear
  in THAT image). Gate: reproduces the pkl's keyframe labels. ~88.8k objects per
  delay step vs 90.5k on keyframes.
* [stale_dataset.py](stages/10_delay_finetune/stale_dataset.py):
  `StaleSweepNuScenesDataset` — each camera independently with p=0.5 swapped for
  its j-th previous sweep, j~U{1,2,3} (≈80–250 ms), honest extrinsics + sweep 2D
  labels; 3D targets stay at keyframe time t. p=0 is bit-identical to the parent
  (control arm). CPU check: sweep 2D centres lifted with honest extrinsics land
  on GT(t) objects at median 0.03 m (naive 0.82 m).
* [ft_stale_r50.py](stages/10_delay_finetune/configs/ft_stale_r50.py): the
  exact base config; backbone frozen (0 trainable), neck + 2D head + 3D head
  train (13.75M of 37.26M); init from the published checkpoint; lr 2e-5, cosine,
  12 epochs = 7380 iters on 2×A100 × bs 2; 62 local train scenes.
* [train_chain.sh](stages/10_delay_finetune/train_chain.sh) (1 h chain),
  [submit_eval.sh](stages/10_delay_finetune/submit_eval.sh),
  [analyze_finetune.py](stages/10_delay_finetune/analyze_finetune.py).

**Setup.** commit `078e7af` (code untracked) · W&B
`khangeldinansar-kaust/multisensor-staleness`, group `stage10_delay_finetune`,
runs `stage10-ft_delay_p0.5` / `stage10-ft_control_p0` · jobs 52828912 (delay),
52828915 (control), each finished in one job (~0.26 s/iter, ~33 min) · seed 0 ·
**pre-registered checkpoint: final iter_7380, no selection on val**.

Pre-registered criteria: delay-FT beats control-FT at honest 76–250 ms (paired,
scene level); and at 0 ms delay-FT loses ≤ 0.005 mAP vs control-FT.

## Result — delay training helps under delay, costs fresh accuracy

Pooled mAP (23 val scenes):

| condition | pretrained | control-FT | delay-FT |
|-----------|-----------:|-----------:|---------:|
| 0 ms (fresh) | 0.4448 | 0.4359 | **0.4202** |
| 76 ms honest | 0.4041 | 0.3996 | **0.4177** |
| 164 ms honest | 0.3290 | 0.3262 | **0.3513** |
| 249 ms honest | 0.2729 | 0.2716 | **0.3013** |
| 76 / 164 / 249 ms naive | 0.380 / 0.287 / 0.224 | 0.372 / 0.283 / 0.225 | 0.392 / 0.302 / 0.235 |
| rr:2 honest (½ compute) | 0.4246 | 0.4199 | 0.4120 |

Paired per-scene, delay-FT vs control-FT (`dumps/analysis_finetune_i7380.json`):

| condition | Δ [95% CI] | wins | sign p |
|-----------|-----------|------|--------|
| 0 ms | **−0.014 [−0.022, −0.007]** | 6/23 | 0.035 |
| 76 ms honest | +0.009 [+0.004, +0.014] | 17/23 | 0.035 |
| 164 ms honest | +0.017 [+0.010, +0.024] | 18/23 | 0.011 |
| 249 ms honest | +0.020 [+0.014, +0.025] | 22/23 | 0.000 |
| 76 / 164 / 249 ms naive | +0.007 / +0.009 / +0.009 (all CI > 0) | 15 / 21 / 20 | |
| rr:2 honest | −0.001 [−0.008, +0.005] | 10/23 | 0.68 |

Control-FT vs pretrained: n.s. in all 8 conditions (|Δ| ≤ 0.003). So
fine-tuning on the 62 local scenes does nothing by itself, and every delay-FT
effect is attributable to the delay augmentation.

**Verdict: mixed — criterion 1 met, criterion 2 failed.**
* Under delay it works, growing with delay: at 249 ms delay-FT wins 22/23 scenes.
  Combined with the honest lift, 76 ms now costs 0.4448 → 0.4177 (−6.1%) instead
  of → 0.3800 (−14.6%) naive: **58% of the 76 ms gap closed**, 35% at 249 ms.
  Training also helps the naive condition, but half as much as honest — geometry
  and training are complementary, not substitutes.
* **Fresh accuracy drops by 0.014–0.016** (gate was 0.005), and rr:2 (half the
  cameras fresh) gains nothing. Delay-FT is almost indifferent to 76 ms delay
  (0.4202 fresh vs 0.4177) — it has become a model that assumes its inputs are
  somewhat late.

## Reading: the model cannot tell WHICH cameras are late

The model has no input saying how old each camera is — the query-side timing is
still hardcoded to (identity, 0), and trained p=0.5 per camera means half its
inputs were late. The best it can do is a hedged correction applied to every
camera: good when all cameras are late, harmful when they are fresh, a wash when
half are (rr:2). This is a prediction, not yet a result: it says a per-camera
age input, trained jointly, should remove the fresh penalty and make rr:2 gain.
It is also the trained version of the "conditioning chimera" question — the
frozen model was hurt by being told the age; a model trained with the age may
use it.

## Caveats

23 val scenes, one seed per arm, 62 train scenes (8.9% of train), frozen
backbone, fixed p=0.5 and 1–3 sweep delays, model run at 2 Hz keyframes.

## Next

* Per-camera age conditioning trained jointly (e.g. an age embedding added to
  each camera's image-token position encoding), same two-arm design; success =
  delay gains kept AND fresh within 0.005 of control AND rr:2 improves.
* Cheap alternative to try first: mode switching — the stack knows the ages, so
  use pretrained weights when all cameras are fresh and delay-FT otherwise.
  Solves uniform delay, not mixed ages.

## Stage 10b: per-camera age conditioning — setup and pre-registered criteria

Tests the stage-10 reading ("the model cannot tell WHICH cameras are late").

* [age_head.py](stages/10_delay_finetune/age_head.py): `AgeAwareStreamPETRHead`
  adds MLP(age / 0.25 s) to the 3D position embedding of each camera's image
  tokens (the keys every query attends to). Age = keyframe timestamp − camera
  capture timestamp (float64; the data side already carries the sweep's true
  capture time under honest geometry, and the ±40 ms intra-keyframe offsets
  otherwise). Last layer zero-initialised in `__init__` → step 0 is EXACTLY the
  pretrained model (unlike the stage-07 port). 33k new params; checkpoint load
  reports exactly the 4 `age_embed` tensors missing. CPU checks: bit-identical to
  the parent head at init; after perturbation each token's shift equals its own
  camera's age embedding (full and top-k paths, max err 4e-7).
* [ft_age_r50.py](stages/10_delay_finetune/configs/ft_age_r50.py): stage-10
  delay config with only the head type changed. One new arm,
  `ft_age_delay_p0.5`; the stage-10 control and delay arms are the comparators.
* Eval: same grid, same pre-registered final checkpoint (iter_7380).
  `eval_async_cams.py` now honours `custom_imports`.

**Pre-registered success (age-delay vs the existing arms, paired per scene):**
1. fresh (0 ms): within 0.005 of control-FT (delay-FT failed this at −0.014);
2. honest 76–250 ms: better than control-FT, and not worse than delay-FT;
3. rr:2 honest (mixed ages): better than control-FT (delay-FT: −0.001, n.s.).
All three → the age input resolves the hedging. 2 without 1/3 → the age signal
is not being used (hedging persists). Note naive-geometry conditions now also
tell the model "age ≈ 0", so they measure the lie, not the delay.

## Stage 10b result — the age input is not used; hedging persists

Eval job 52885297 (8 tasks), final checkpoint iter_7380 as pre-registered.
Paired per-scene tests in
[analysis_finetune_age_i7380.json](stages/10_delay_finetune/dumps/analysis_finetune_age_i7380.json)
(`analyze_finetune.py` now also compares the age arm; the stage-10-only file is
unchanged).

| condition | control-FT | delay-FT | age-FT | age vs control | age vs delay |
|-----------|-----------:|---------:|-------:|----------------|--------------|
| 0 ms (fresh) | 0.4359 | 0.4202 | **0.4137** | **−0.016 [−0.023, −0.010], 2/23** | −0.002 [−0.005, −0.000] |
| 76 ms honest | 0.3996 | 0.4177 | 0.4134 | +0.008 [+0.003, +0.013], 17/23 | −0.001 n.s. |
| 164 ms honest | 0.3262 | 0.3513 | 0.3513 | +0.017 [+0.011, +0.024], 20/23 | +0.001 n.s. |
| 249 ms honest | 0.2716 | 0.3013 | 0.3001 | +0.019 [+0.013, +0.025], 22/23 | −0.001 n.s. |
| rr:2 honest | 0.4199 | 0.4120 | **0.4047** | −0.002 [−0.009, +0.003], n.s. | −0.001 n.s. (5/23) |
| 76/164/249 naive | 0.372/0.283/0.225 | 0.392/0.302/0.235 | 0.391/0.299/0.234 | all +0.008–0.010, sig. | all n.s. |

Against the pre-registered criteria: **(1) fresh within 0.005 of control —
FAILED** (−0.016, worse than delay-FT's −0.014); **(2) delay gains kept — met**
(better than control, indistinguishable from delay-FT); **(3) rr:2 better than
control — FAILED** (n.s., point estimate negative). By the pre-registered
reading, "2 without 1/3 → the age signal is not being used (hedging persists)".

**Verdict: didn't work.** The age arm is statistically indistinguishable from
delay-FT in every one of the 8 conditions. Decisive tell: in the naive
conditions the model is told age ≈ 0, yet it gains exactly as much as delay-FT
there — it behaves the same whether or not it is told the truth, i.e. it ignores
the age input and applies the same hedged correction to every camera.

Wiring was checked, not assumed: timestamps reach the head as float64; the
checkpoint holds trained `age_embed` weights (last layer moved off its zero
init); eval builds `AgeAwareStreamPETRHead` via `custom_imports` with no
missing/unexpected keys. The embedding did grow — ‖emb(250 ms) − emb(0)‖ = 1.73,
‖emb(78 ms) − emb(0)‖ = 0.56 — so "the module never trained" is ruled out.
Whether that is large relative to the position embedding it is added to is not
measured (needs a forward pass).

**Live explanations, most mundane first:**
1. **Learning rate.** One shared lr 2e-5 for a zero-initialised new module; the
   pretrained layers can absorb the delay by hedging faster than the new input
   grows. Test: 10–100× lr on `age_embed` only (mmcv `paramwise_cfg`).
2. **Placement.** Added to the key position embedding only; the query side still
   gets (identity, dt=0), and StreamPETR's topk query spawning is driven by the
   2D ROI head, which never sees age.
3. **Data.** 62 scenes, one seed per arm.

This is the second time an explicit age input failed to help (stage 07 port,
frozen model; now a trained model with a zero-init input). The difference from
stage 07 is that here the input starts harmless and training still chose to
ignore it — so "the conditioning chimera" is no longer the explanation; the
model simply finds hedging easier than reading the age.

**What stands from stages 09–10:** honest geometry + delay fine-tuning under
uniform delay (76 ms: −6.1% instead of −14.6%). **What does not work yet:**
getting a model that is both fresh-accurate and delay-robust. The cheap
deployable fix remains mode switching (pretrained weights when all cameras are
fresh, delay-FT otherwise), which handles uniform delay but not mixed ages.

## Stage 10c: lr sweep on the age embedding — setup and pre-registered criteria

Tests the most mundane explanation for 10b: the zero-initialised `age_embed`
shared lr 2e-5 with the pretrained neck/heads. Under AdamW a weight moves at most
~lr per step, so over 7380 cosine iters it could drift ~0.1; its last layer
reached 0.02 while the pretrained layers absorbed the delay by hedging.

* [ft_age_lr_r50.py](stages/10_delay_finetune/configs/ft_age_lr_r50.py):
  ft_age_r50.py + `paramwise_cfg.custom_keys['age_embed'].lr_mult = AGE_LR_MULT`
  (env). Everything else identical (data, p=0.5 delay, schedule, init, seed 0).
  CPU check of the built optimizer at ×30: `age_embed` lr 6e-4, all other
  trainable params 2e-5, base `img_backbone` key preserved.
* Three arms: ×10 / ×30 / ×100 (age lr 2e-4 / 6e-4 / 2e-3). W&B runs
  `stage10-ft_age_delay_p0.5_lrx{10,30,100}`, group `stage10_delay_finetune`.
* Eval: same 8-condition grid, final checkpoint iter_7380 (no selection).

**Pre-registered:** the three 10b criteria (fresh within 0.005 of control;
honest 76–250 ms better than control and not worse than delay-FT; rr:2 better
than control), checked per lr. Because three lrs are tried, a success also
requires the age arm to beat **delay-FT** (paired, CI > 0) at fresh or rr:2 —
the conditions where knowing the age is the only possible advantage. Report all
three lrs regardless. Diagnostic per run: ‖emb(250 ms) − emb(0)‖ (10b: 1.73).
If all three fail: the age signal is not under-trained but unused, and
explanations 2 (placement: key side only, ROI spawning blind to age) and 3
(data) remain.

## Stage 10c result — a 6–15× stronger age signal changes nothing

Eval: 24 tasks, final checkpoint iter_7380 for each lr. Paired per-scene tests in
[analysis_finetune_agelr_i7380.json](stages/10_delay_finetune/dumps/analysis_finetune_agelr_i7380.json).
Training healthy: loss curves match 10b; the 4 NaN log lines per run also occur
in 10b (fp16 dynamic loss scale), not divergence.

The embedding did grow — ‖emb(250 ms) − emb(0)‖ = 1.7 (10b) → 10.0 / 22.1 / 26.7
at ×10 / ×30 / ×100; ‖emb(78 ms) − emb(0)‖ = 0.6 → 3.2 / 6.8 / 8.0.

| condition | control | delay-FT | ×10 | ×30 | ×100 |
|-----------|--------:|---------:|----:|----:|-----:|
| 0 ms (fresh) | 0.4359 | 0.4202 | 0.4157 | 0.4183 | 0.4183 |
| 76 ms honest | 0.3996 | 0.4177 | 0.4043 | 0.4075 | 0.4118 |
| 164 ms honest | 0.3262 | 0.3513 | 0.3507 | 0.3531 | 0.3542 |
| 249 ms honest | 0.2716 | 0.3013 | 0.2985 | 0.3014 | 0.3010 |
| rr:2 honest | 0.4199 | 0.4120 | 0.4090 | 0.4104 | 0.4086 |

* **vs control**, every lr: fresh −0.014 [CI excludes 0] (3–5/23 wins); honest
  76/164/249 ms +0.007 / +0.018 / +0.020 (sig.); rr:2 −0.001 (n.s.).
* **vs delay-FT**: **24 of 24 comparisons n.s.**, all |Δ| ≤ 0.0023.

Against the pre-registered criteria, at every lr: (1) fresh within 0.005 of
control — FAILED; (2) delay gains — met; (3) rr:2 better than control — FAILED;
and the added requirement (beat delay-FT at fresh or rr:2) — FAILED.

**Verdict: didn't work. Under-training is ruled out as the cause.** An age
signal 6–15× larger than 10b's leaves the model indistinguishable from the
model that was never told the age, to within ±0.002 mAP.

**Likely mechanism (hypothesis, not tested):** the embedding is added to the
position encoding of the KEYS only. A vector added to every key of one camera
shifts the attention logits of that camera's tokens by q·c — it can only
re-weight how much attention a camera gets, it cannot move WHERE that camera's
evidence says an object is, and values are unchanged. Down-weighting a stale
camera does not help when it is the only view of that region, which is most of
the surround. So a large signal at this site has nothing useful to do, and the
model ignores it. Consistent with the 10b observation that naive conditions
(age told = 0) gain exactly as much as honest ones.

**Where this leaves "train for delay":** the useful thing training does is a
camera-agnostic hedge (stage 10). Telling the model per-camera age at the key
side does not let it un-hedge on fresh cameras. Getting fresh accuracy back
needs either the age where it can change geometry/content (query reference
points, values, or the ROI spawn head), or no learning at all — mode switching
between pretrained and delay-FT weights, which the stack can do because it
knows the ages.

## Stage 10d: mode switching — setup and pre-registered criteria

The stack knows every camera's age, so instead of one model that must be good at
both, run the pretrained weights when all cameras are fresh and delay-FT
otherwise. No training.

**Constant delay (exact, from existing dumps).** The switch is constant within a
run, so it is a per-condition pick: fresh → pretrained 0.4448; honest
76 / 164 / 249 ms → delay-FT 0.4177 / 0.3513 / 0.3013, each significantly above
pretrained (+0.009 / +0.015 / +0.019 per scene). So under constant delay the
switch dominates both single models: pretrained's fresh accuracy, delay-FT's
delayed accuracy. Mixed ages are where it does not help: rr:2 (half the cameras
fresh) has max age > 0, so the rule picks delay-FT 0.4120 vs pretrained 0.4246
(−0.004 n.s.).

**Delay that changes frame to frame (the real test, GPU).** Weights then swap
mid-scene and each model reads temporal memory written by the other — the same
hazard as the Phase-4 memory chimera. Built:
* `rand:J` schedule in
  [eval_async_cams.py](stages/09_subframe_async/eval_async_cams.py): each frame
  all cameras share one age U{0..J}, seeded by sample token. rand:3 = 28% fresh
  frames, mean 119 ms.
* `--switch-ckpt / --switch-above-ms`: swaps the 267 differing tensors before a
  frame when needed; memory is carried across swaps (it lives in attributes, not
  the state_dict). Frames are matched to infos by loader order, checked per
  frame against scene_token + first-camera filename. CPU check on 40 real
  pipeline frames: weights equal the chosen mode every frame; an induced desync
  raises.
* [submit_switch.sh](stages/10_delay_finetune/submit_switch.sh): rand:3 honest,
  4 tasks — pretrained only, delay-FT only, switch at >0 ms (pre-registered
  rule), switch at >100 ms (exploratory).

**Pre-registered:** the >0 ms switch must beat BOTH single models on rand:3
(paired per scene, CI > 0). Diagnostic: compare with the splice of the two
single-model runs (each frame's detections from the model the rule picks, no
cross-model memory) — the gap between splice and the real switch is the price of
reading the other model's memory.

## Stage 10d result — switching works, and cross-model memory is a gain, not a chimera

Job 53829711 (4 tasks), rand:3 honest (252 fresh / 662 stale frames; the >0 ms
switch swapped weights 366 times). Analysis:
[analyze_switch.py](stages/10_delay_finetune/analyze_switch.py) →
`dumps/analysis_switch_rand3.json`; fresh/stale paired split (exploratory, after
seeing the pooled table) → `dumps/analysis_switch_rand3_by_freshness.json`.

| run | all | fresh frames | stale frames |
|-----|----:|-------------:|-------------:|
| pretrained only | 0.3683 | 0.4150 | 0.3547 |
| delay-FT only | 0.3908 | 0.4194 | 0.3819 |
| **switch >0 ms** | **0.3957** | **0.4394** | **0.3838** |
| switch >100 ms | 0.3932 | 0.4336 | 0.3811 |
| splice >0 ms (no cross-model memory) | 0.3856 | 0.4150 | 0.3819 |
| splice >100 ms | 0.3805 | 0.4150 | 0.3733 |

Paired per scene (n=23):

| comparison | Δ [95% CI] | wins | |
|------------|-----------|------|-|
| switch0 vs pretrained | +0.016 [+0.009, +0.024] | 20/23 | SIG |
| **switch0 vs delay-FT** (pre-registered) | +0.002 [−0.001, +0.004] | 16/23 | n.s. |
| switch100 vs pretrained / delay-FT | +0.015 SIG / +0.001 n.s. | 18 / 14 | |
| splice0 vs switch0 | **−0.004 [−0.006, −0.001]** | 9/23 | SIG |
| splice100 vs switch100 | **−0.006 [−0.009, −0.003]** | 5/23 | SIG |
| *fresh frames:* switch0 vs delay-FT | **+0.008 [+0.004, +0.011]** | 18/23 | SIG (exploratory) |
| *stale frames:* switch0 vs delay-FT | +0.000 [−0.003, +0.003] | 13/23 | n.s. |

**Verdict: mixed — pre-registered criterion NOT met, mechanism clear.** The
switch beats pretrained everywhere, but over the whole sequence it is not
significantly better than simply running delay-FT always (+0.002, CI touches 0).

What the decomposition shows:
1. **No memory chimera — the opposite.** The real switch beats the splice of the
   two single-model runs on fresh frames by +0.024 pooled (0.4394 vs 0.4150), and
   overall significantly in both thresholds. The pretrained weights on a fresh
   frame do BETTER reading memory that delay-FT wrote on preceding stale frames
   than memory they wrote themselves. Unlike LTC (Phase 4), swapping the model
   that writes memory is benign here — both writers share one memory format.
2. **Stale frames poison later fresh frames.** Pretrained-only on rand:3 scores
   0.4150 on its fresh frames vs 0.4448 when every frame is fresh: the cost of
   delay is not confined to the late frame, it is carried forward in memory.
   Delay-FT writes better memory under delay, which is where the switch's
   fresh-frame gain comes from.
3. **The gain is all on fresh frames** (+0.008 vs delay-FT, 18/23, exploratory);
   on stale frames the switch is delay-FT by construction (n.s., as expected).
   With only 28% fresh frames it dilutes to a non-significant +0.002 overall —
   so the benefit of switching scales with the fraction of fresh frames.
4. The >100 ms threshold is no better than >0 ms (uses pretrained on 76 ms
   frames, which delay-FT handles better).

**Deployable statement:** with honest geometry, run delay-FT whenever any camera
is late and the pretrained weights when all are fresh. Under constant delay this
recovers pretrained fresh accuracy and delay-FT delayed accuracy exactly; under
frame-to-frame jitter it costs nothing versus delay-FT always and gains on fresh
frames. Caveats: one jitter pattern (U{0..3}, 28% fresh), the stack's age
knowledge is assumed exact, 23 scenes.

## Stage 10e: does the switch gain grow with the fresh-frame share? — pre-registered

Stage 10d reading: the switch's gain over delay-FT lives on fresh frames, so it
should grow with their share. Test: same design at rand:1 (each frame age 0 or
~76 ms; ~half the frames fresh) — `SCHEDULE=rand:1 sbatch --array=0-2
submit_switch.sh` (pretrained, delay-FT, switch >0 ms); analysis
`analyze_switch.py --schedule rand1` (now also does the fresh/stale split that
was exploratory in 10d).
**Prediction:** switch − delay-FT over all frames is larger than at rand:3
(+0.002) and significant (paired per scene, CI > 0); on fresh frames still
positive.

## Stage 10e result — the prediction holds: more fresh frames, larger switch gain

Job 53959868 (3 tasks), rand:1 honest: 465 fresh / 449 stale (~76 ms) frames,
442 weight swaps. `dumps/analysis_switch_rand1.json`.

| run | all | fresh frames | stale frames |
|-----|----:|-------------:|-------------:|
| pretrained only | 0.4227 | 0.4305 | 0.4219 |
| delay-FT only | 0.4135 | 0.4173 | 0.4140 |
| **switch >0 ms** | **0.4262** | **0.4329** | **0.4247** |
| splice (no cross-model memory) | 0.4208 | 0.4305 | 0.4140 |

| comparison (paired, n=23) | Δ [95% CI] | wins | |
|---------------------------|-----------|------|-|
| **switch vs delay-FT** | **+0.011 [+0.005, +0.017]** | **19/23** | SIG (sign p=0.003) |
| switch vs pretrained | +0.005 [+0.001, +0.008] | 13/23 | CI > 0, sign test n.s. |
| splice vs switch | −0.003 [−0.005, −0.001] | 6/23 | SIG |
| fresh frames: switch vs delay-FT | +0.014 [+0.008, +0.021] | 18/23 | SIG |
| stale frames: switch vs delay-FT | +0.006 [+0.001, +0.013] | 15/23 | CI > 0 |
| stale frames: switch vs pretrained | +0.009 [+0.004, +0.015] | 16/23 | CI > 0 |

**Verdict: worked — pre-registered prediction confirmed.** Switch − delay-FT grew
from +0.002 (n.s., 28% fresh) to **+0.011 (19/23, 51% fresh)**. At rand:1 the
switch also edges out the pretrained model (CI > 0, sign test n.s.), so it is
the best single policy at both jitter levels, and unlike either fixed model it
never loses badly: at rand:3 pretrained is far behind, at rand:1 delay-FT is.

The memory finding replicates and is now visible in both directions: the switch
beats the splice again, and on stale frames the delay-FT weights do better
reading memory written by the pretrained model on fresh frames (+0.006 vs
delay-FT alone). Mixed-writer memory is consistently a gain.

## Stages 09–10: closing summary

* Realistic delay is costly: one ~80 ms frame on all cameras = −14.7% mAP under
  the synchronous interface.
* Honest geometry (zero parameters) closes 38% of that at 80 ms, 16% at 500 ms;
  halves the price of skipping camera backbone passes (½ compute: −4.5% vs −7.6%).
* Delay-augmented fine-tuning adds more under delay (76 ms: −6.1%) but costs
  fresh accuracy (−0.014); explicit per-camera age on attention keys is ignored,
  even with a 6–15× stronger signal (10b/10c).
* **Mode switching resolves the trade-off without new training:** exact best-of-
  both under constant delay; under frame-to-frame jitter it is never worse than
  either fixed model and its gain over delay-FT grows with the fresh-frame share
  (+0.002 n.s. at 28% → +0.011 sig. at 51%). Swapping weights mid-scene does
  not create a memory chimera — mixed-writer memory helps.
* Open: per-camera mixed ages (rr:2) are still handled best by the pretrained
  weights alone; all results on 23 val scenes, one seed, 62 train scenes.

# Stage 08 after the audit fixes

Code fixes (all CPU, outputs `stages/08_risk_envelopes/dumps/*_fixed*`, run by
[rerun_fixed.sh](stages/08_risk_envelopes/rerun_fixed.sh); originals untouched):

* `conformal_envelopes.py` — per class: number of splits with q = inf and the
  median over finite splits (a median of partly-inf splits is not a radius);
  `cov_bounded` (objects in q = inf classes counted as NOT covered); held-out
  coverage per class (where q is finite) and per speed stratum (was in-sample);
  claimed area priced on each split's TEST scenes, eligible frames only (was all
  frames); the ρ sign test is labelled a consistency check (overlapping splits).
* `envelope_coverage.py` — area raster excludes the first-k fresh-fallback frames,
  fresh and stale on the same frames.
* `envelope_precision.py` — detections from fresh-fallback frames excluded from
  precision and envelopes/frame; frame stride now applied over the allowed frames
  (striding first could leave a group with no sampled frame → area 0).
* `refresh_policy.py` — calibration excludes frames with index < k; the
  zero-area-group bug is fixed by the stride change; `random` averaged over 3
  seeds; 4 area splits (was 2), per-split areas saved for paired comparison.

## Corrected numbers (d1_key = 500 ms, honest key lift, α = 0.1)

**Calibrated radius per class** (`conformal_d1_key_fixed_noarea.log`):

| class | inf splits | q (finite splits) | held-out coverage (finite q) |
|-------|-----------:|------------------:|-----------------------------:|
| car | 0/40 | 4.31 m | 0.897 |
| pedestrian | 0/40 | 3.80 m | 0.896 |
| traffic_cone | 0/40 | 2.20 m | 0.880 |
| barrier | 0/40 | 4.85 m | 0.867 |
| truck | 0/40 | 10.12 m | 0.893 |
| bicycle | 10/40 | 30.80 m | 0.884 |
| bus | 19/40 | 4.33 m | **0.717** |
| trailer | 20/40 | 6.14 m | 0.875 |
| motorcycle | 24/40 | 29.48 m | **0.608** |
| construction_vehicle | 28/40 | 12.70 m | **0.641** |

* Deployable bounds hold for car, pedestrian, cone (and truck at 10 m).
* Trailer is NOT "no bound": it is inf only when calibration has < 9 trailers;
  when it has enough, 6.14 m covers 0.875 held-out.
* Bus, motorcycle, construction vehicle: finite radii do not transfer to unseen
  scenes (0.61–0.72 held-out) — no usable bound on 23 scenes; an explicit "no
  bound" output is the honest answer for them.
* Achieved coverage 0.894 counts the 2.9% of test objects in q = inf classes as
  covered; counting them uncovered gives **0.866**. α table (v3 values are the
  correct ones): 0.797 / **0.894 (40% of splits ≥ nominal)** / 0.942 at α =
  0.2 / 0.1 / 0.05.
* Speed strata, held out (β = 0): 0.893 / 0.892 / 0.904 / 0.888 — still no
  conditional failure; β = 0.5 still costs 1.65× car area.
* Anisotropy: car area 0.0885 (disc) → 0.0844 at ρ = 1.25 (−4.7%, 8/8 splits —
  overlapping splits, consistency not significance); ρ = 3 is +15%.

**Cost of 500 ms at equal coverage, cars, budget 50** (`precision_*_fixed`):
stale q 3.90 m vs fresh 1.86 m (**2.10× radius**); claimed area 0.0763 vs
0.0315 (**2.42× area**; was 2.36×). Precision 0.325 vs 0.255 — unchanged
reading (precision is the detector's, and not comparable across radii). Budget:
stale area is flat from 30 to 300 (0.0763–0.0790); fresh minimum at 50.

**Refresh policy** (`refresh_policy_fixed.*`, cars, budget 50):

| refresh rate | 1.00 | 0.50 | 0.35 |
|--------------|-----:|-----:|-----:|
| claimed area (periodic) | 0.0315 | 0.0518 | 0.0808 |

Rate 1.0 now equals the fresh detector's area (0.0315), resolving the earlier
0.0235 vs 0.0302 mismatch. Halving refreshes costs 1.65× area.

Against periodic at the same rate (per-split chord,
`refresh_policy_fixed_vs_periodic.json`, 4 overlapping splits — indicative only):

| policy family | area vs periodic | splits better |
|---------------|-----------------:|---------------|
| random (3 seeds) | **+11 to +18%** | 0/4 at every rate |
| ego trigger | −3 to −11% | 3–4/4 |
| obj trigger | 0 to −5% | mixed |

So scheduling matters (random is worst everywhere, now averaged over seeds), and
the ego trigger is modestly better than periodic at mid–high rates — the earlier
"no policy beats periodic" was too strong, but the gain is small and not tested
for significance. Radius per (k, ego bucket) at k=1: 3.11 / 2.51 / 3.32 / 4.62 m
(was 3.48 at < 0.5 m; still non-monotone, less so).

Unchanged limits: k ≤ 3 clamp (no refresh rate below 0.346), and the k-dumps are
"always k-stale" runs, not a policy simulated in the loop.
