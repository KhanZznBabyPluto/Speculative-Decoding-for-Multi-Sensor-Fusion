#!/usr/bin/env bash
# Install mmdet3d 1.0.0rc6 from source INSIDE StreamPETR/.
# Run from /ibex/user/khangea/multisensor/StreamPETR after activating
# the streampetr_env conda env.
#
# StreamPETR ships against this specific mmdet3d tag — using our root-level
# mmdetection3d (which is 1.4.0) will not work.

set -e

cd /ibex/user/khangea/multisensor/StreamPETR

if [ ! -d mmdetection3d ]; then
    git clone https://github.com/open-mmlab/mmdetection3d.git
fi
cd mmdetection3d
git checkout v1.0.0rc6
pip install -e .
echo
echo "=== mmdet3d 1.0.0rc6 installed (editable) inside StreamPETR/ ==="
