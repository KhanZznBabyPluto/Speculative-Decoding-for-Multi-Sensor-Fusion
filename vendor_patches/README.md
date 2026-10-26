# vendor_patches/

The three third-party repos are git clones kept out of this project's git (see
`.gitignore`). Every local change we made to them is captured here as a patch, so
the clones can be re-created exactly:

```
git clone <url> <dir> && git -C <dir> checkout <commit> && git -C <dir> apply ../vendor_patches/<dir>.patch
```

| dir | upstream | commit | patch | used by |
|-----|----------|--------|-------|---------|
| `StreamPETR/` | https://github.com/exiawsh/StreamPETR.git | `95f6470` | `StreamPETR.patch` (4 files) | stages 04, 07–10 (`streampetr_env`) |
| `StreamPETR/mmdetection3d/` | https://github.com/open-mmlab/mmdetection3d.git | `47285b3f` (v1.0.0rc6) | none (unmodified) | editable install of `streampetr_env` — do not move |
| `MamBEV/` | https://github.com/amai-gsu/MamBEV.git | `979f77a` | `MamBEV.patch` (7 files) | stages 05–06 (`mambev_env`, which uses pip mmdet3d 1.4.0) |
| `mmdetection3d/` | https://github.com/open-mmlab/mmdetection3d.git | `fe25f7a5` | `mmdetection3d.patch` (1 file) + two empty `projects/**/__init__.py` | editable install of the `sensor` env (archived stages 01–03) — do not move |

What the patches do (each hunk is marked `# PATCH (stage NN)` in the source):

* **StreamPETR** — guard `flash_attn` imports (`models/utils/attention.py`);
  converter: val-only by default, `STREAMPETR_INCLUDE_TRAIN=1` to emit train
  infos (stage 07), skip samples whose blobs are absent (partial download);
  `create_data_nusc.py` skips v1.0-test; `tools/test.py` re-enables the
  single-GPU path.
* **MamBEV** — dead import removed in `hydra/__init__.py`; stale
  `base_hydra_block` → `base_block` imports fixed in the conv-gate ablation
  blocks and mixins.
* **mmdetection3d** (top level) — converter skips samples whose lidar file is not
  in the downloaded blob.

Regenerate after any further vendor edit:

```
for r in StreamPETR MamBEV mmdetection3d; do git -C $r diff -- . ':(exclude)*.pyc' > vendor_patches/$r.patch; done
```
