# CLAUDE.md — multisensor (StreamPETR staleness / async perception)

Project facts only; global working rules live in ~/.claude/CLAUDE.md.
Start from the "READ ME FIRST" table at the top of `NOTES.md` (lab notebook,
append-only; later sections correct earlier ones).

## Layout

* `stages/04..10_*` — live code, one flat dir per stage (scripts + `dumps/` +
  `logs/`). `archive/` — dead stages 01–03 and superseded docs (not imported).
* `data_full/` — nuScenes dataroot (partial download: 23 val scenes / 914
  samples; 12 Hz camera sweeps present). `data_withtrain/` — pkls incl. 62
  train scenes (symlinks into data_full). `checkpoints/` — StreamPETR R50 +
  `mambev/`.
* `StreamPETR/`, `MamBEV/`, `mmdetection3d/` — vendored clones, not in git;
  our edits are in `vendor_patches/` (see its README).

## Searching — keep vendored code out of context

* `.ignore` excludes vendored repos, data, checkpoints and archive from Grep/Glob.
  With shell grep/find, pass `stages/` (or a specific dir), never `.` or the
  vendored trees. Do not send Explore agents into vendored repos.
* Open a vendored file only for a fact not listed below, and then read the
  specific lines, not the module.

## Environments

* `streampetr_env` — mmdet3d 1.0.0rc6, editable from `StreamPETR/mmdetection3d`
  (never move it). flash-attn ⇒ GPU jobs need `gpu:a100`. Run from
  `StreamPETR/` with `PYTHONPATH=$PWD`. Dataset/model building and the plugin
  import work on CPU (only inference needs CUDA).
* `mambev_env` — pip mmdet3d 1.4.0; MamBEV plugin import needs CUDA, but
  `Config.fromfile(..., import_custom_modules=False)` works on CPU.
* `sensor` — archived stages 01–03; editable install of top-level `mmdetection3d/`.
* conda: `source /ibex/user/khangea/miniforge/etc/profile.d/conda.sh`;
  envs under `/ibex/user/khangea/conda-environments/`.

## StreamPETR facts (verified in source — no need to re-read)

* `data['ego_pose']` = lidar→global 4×4 (`e2g @ l2e`), plus `ego_pose_inv`.
* After `post_update_memory` (`streampetr_head.py:~372`):
  `memory_reference_point` is GLOBAL (metres); `memory_velo` is the raw
  regression output in lidar(t), never rotated — rotate by `ego_pose[:3,:3]`
  before combining. Memory = 256 new (propagated) + 1024 carried tokens, stored as
  plain attributes (not in `state_dict`, so weight swaps keep memory).
* Image-born queries get `rec_ego_pose = eye(4)` and a zero time embedding
  (head ~402–438): the synchrony assumption.
* `get_data_info` builds `lidar2img` from `info['cams'][cam]`:
  `data_path`, `sensor2lidar_rotation/translation`, `cam_intrinsic`,
  `timestamp`. Camera order: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT,
  BACK_RIGHT. Rewriting these fields in `dataset.data_infos` is how stage 09
  feeds stale sweeps (no model hooks).
* `collect_keys`: lidar2img, intrinsics, extrinsics, timestamp, img_timestamp
  (both float64 through the pipeline), ego_pose, ego_pose_inv.
* Test `img_metas` carry NO sample token (keys incl. `scene_token`, `filename`);
  the test loader is sequential and unshuffled — index frames by loader order.
* Entry points: `Petr3D.simple_test(img_metas, **data)` → `extract_img_feat`.
* Converter: `obtain_sensor2top` gives each camera its own capture-time pose
  (honest extrinsics for any sample_data, incl. sweeps); GT velocity is stored
  in the lidar frame. Val pkl ego pose is split fields (`ego2global_rotation`
  quaternion + translation), no 4×4 key.
* Config `projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py`,
  ckpt `checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth`; fresh mAP on our
  subset 0.4448 (every harness gate checks this).

## Evaluation conventions

* nuScenes GT must be filtered to our 914 tokens:
  `measure_stale_map._install_partial_blob_eval_patch` (GPU scripts) or
  `stages/04_streampetr_latent/eval_subset.SubsetEvaluator` (CPU re-eval of saved
  `results_nusc.json`).
* The independent unit is the SCENE (n=23): use `bootstrap_ci.per_scene_map` +
  `paired_tests`; never quote a delta without them. First k frames of a scene
  fall back to fresh — exclude them for stale-only numbers.
* Never overwrite results: new outputs get new names (`*_fixed`, `*_v2`).
* Loading NuScenes metadata takes ~1 min and several GB; run at most one or two
  such processes at once (parallel loads were OOM-killed). Detach long CPU runs
  with `setsid nohup ... &` — background shells die with the session.
* MamBEV config traps (handled in `stages/06_gated_bev_psm/measure_mambev_baseline.py`
  `prepare_cfg`): `indices=500`, `WandbVisBackend`, missing `init_cfg` path.
* W&B: entity `khangeldinansar-kaust`, project `multisensor-staleness`.
