#!/usr/bin/env bash
# Stage 05 — MamBEV environment setup.
#
# Why a separate env (not `sensor`):
#   * MamBEV needs Python 3.10, PyTorch 2.3.1 + CUDA 12.1, and mamba-ssm 2.2.2
#     (+ causal-conv1d 1.4.0). `sensor` is Python 3.11 with a different torch /
#     cuda combo and no mamba-ssm — mixing risks breaking stage 03.
#   * MamBEV itself uses MODERN mmcv 2.1 / mmdet 3.3 / mmdet3d 1.4 / mmengine
#     0.10 — so this is structurally similar to `sensor`, just with different
#     pins. NOT the legacy mmcv-1.x stack like StreamPETR.
#
# Source of truth: [MamBEV/env_cfg/torch231.yml](../../MamBEV/env_cfg/torch231.yml)
# + [MamBEV/env_cfg/install_env.sh](../../MamBEV/env_cfg/install_env.sh).
# We re-derive here instead of `conda env update -f torch231.yml` because
# (a) we want the env at /ibex/user/khangea/conda-environments/ to match
# `streampetr_env` / `sensor` convention, (b) the yml hard-pins exact
# build hashes that may not resolve on this cluster's channels, and (c) we
# want each pip install to be auditable for debugging.
#
# Run on a GPU node (mamba-ssm + causal-conv1d may fall back to compiling
# from source which needs nvcc). ~30-60 minutes.

set -e

ENV_NAME=mambev_env
ENV_PREFIX=/ibex/user/khangea/conda-environments/${ENV_NAME}
MAMBEV_ROOT=/ibex/user/khangea/multisensor/MamBEV

# --- 1. Create env ----------------------------------------------------------
conda create -p "${ENV_PREFIX}" python=3.10 -y
source activate "${ENV_PREFIX}"

# --- 2. PyTorch 2.3.1 + CUDA 12.1 ------------------------------------------
#       NOTE: we install torch via PIP, not conda. Reason: on this cluster's
#       conda config, `conda install -c pytorch -c nvidia pytorch==2.3.1`
#       silently solves to the conda-forge `cpu_generic` build (channel
#       priority loses to conda-forge). Using pip's pytorch index pins
#       cu121 unambiguously.
#       cuda-nvcc (for compiling mamba-ssm in step 4) comes from a separate
#       conda install so it's discoverable on PATH at build time.
conda install -y -c nvidia/label/cuda-12.1.1 cuda-toolkit
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# --- 3. Toolchain prerequisites for from-source builds ---------------------
#       mmcv 2.1.0, causal-conv1d, mamba-ssm all compile from source on this
#       torch/cuda combo. Three foot-guns from one painful session:
#         (a) setuptools >= 81 removed `pkg_resources` → mmcv setup.py breaks.
#         (b) pip's default build-isolation env has no torch → these setup.py
#             files `import torch` at module load and explode.
#         (c) torch's cpp_extension queries `nvcc --version` via PATH; the
#             cluster's system PATH points at cuda 11.8 which mismatches the
#             torch 2.3.1+cu121 wheel and aborts the build. CUDA_HOME alone
#             is NOT enough — PATH must put conda's nvcc first.
#       We solve (a) by pinning setuptools, (b) by `--no-build-isolation` on
#       every from-source install below, (c) by exporting CUDA_HOME + PATH.
pip install "setuptools<81" wheel ninja

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export MAX_JOBS=$(nproc)                  # parallel compile (~10× faster)
export TORCH_CUDA_ARCH_LIST="8.0;9.0"     # A100 + H100; adjust to hardware

# --- 4. OpenMMLab stack -----------------------------------------------------
pip install mmcv==2.1.0 --no-build-isolation \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
pip install mmengine==0.10.4 mmdet==3.3.0 mmdet3d==1.4.0

# --- 5. Mamba kernels -------------------------------------------------------
#       Same --no-build-isolation reasoning. mamba-ssm imports torch at
#       setup time to query CUDA version.
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.2     --no-build-isolation

# --- 6. detectron2 (from source; no PyPI wheel for torch 2.3) --------------
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# --- 7. Other MamBEV pip deps (subset of torch231.yml, pinned identically) -
pip install \
    nuscenes-devkit==1.1.11 \
    numpy==1.26.4 \
    pyquaternion==0.9.9 \
    einops==0.8.0 \
    numba==0.60.0 \
    plyfile==1.0.3 \
    shapely==1.8.5.post1 \
    scikit-image==0.24.0 \
    scikit-learn==1.5.1 \
    scipy==1.14.0 \
    opencv-python==4.10.0.84 \
    pandas==2.2.2 \
    matplotlib==3.5.0 \
    tensorboard==2.17.0 \
    yapf==0.40.2 \
    fvcore==0.1.5.post20221221 \
    iopath==0.1.9 \
    lyft-dataset-sdk==0.0.8 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    huggingface-hub==0.24.3 \
    transformers==4.43.3 \
    tokenizers==0.19.1 \
    safetensors==0.4.3 \
    addict==2.4.0 \
    terminaltables==3.1.10 \
    tabulate==0.9.0 \
    portalocker==2.10.1 \
    descartes==1.1.0 \
    trimesh==4.4.3 \
    ninja==1.11.1.1 \
    seaborn==0.13.2

# --- 8. Register MamBEV's projects.* package -------------------------------
#       The upstream recipe says `pip install -e .` is enough. It isn't, on
#       this codebase: MamBEV's setup.py uses `find_packages(include=
#       ['projects*'])`, but `projects/` and `projects/MAMBEV/` have no
#       `__init__.py`. Without them, find_packages() returns []. The
#       editable install completes but exposes nothing — every
#       `projects.X.Y` import fails with ModuleNotFoundError.
#       Workaround: drop a .pth file in site-packages so MamBEV root joins
#       sys.path at interpreter startup. Python 3.3+ then treats `projects`
#       as a namespace package and the sub-imports resolve fine.
if [ ! -d "${MAMBEV_ROOT}" ]; then
    echo "ERROR: MamBEV repo not at ${MAMBEV_ROOT}. Clone first." >&2
    exit 1
fi
cd "${MAMBEV_ROOT}"
pip install -e .
cd -
echo "${MAMBEV_ROOT}" > "${ENV_PREFIX}/lib/python3.10/site-packages/mambev_root.pth"

# --- 9. Smoke-test imports --------------------------------------------------
python - <<'PY'
import importlib
for mod in (
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmdet3d",
    "mamba_ssm",
    "causal_conv1d",
    "projects.MAMBEV.mambev",
    "projects.BEVFormer.bevformer",
    "projects.PETR.petr",
    "projects.DETR3D.detr3d",
):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "n/a")
        print(f"  OK  {mod:<35}  {ver}")
    except Exception as exc:
        print(f"  XX  {mod:<35}  {type(exc).__name__}: {exc}")
PY

echo
echo "=== Environment ${ENV_NAME} created at ${ENV_PREFIX} ==="
echo "Activate:  source activate ${ENV_PREFIX}"
echo "Next:      cd stages/05_mambev_dense_bev && python verify_checkpoint.py"
