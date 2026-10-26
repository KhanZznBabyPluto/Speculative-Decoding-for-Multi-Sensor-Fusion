"""
StreamPETR head that is told how old each camera's image is.

`AgeAwareStreamPETRHead` adds a per-camera age embedding to the 3D position
embedding of that camera's image tokens (the keys/values every query attends
to). Age = keyframe timestamp - camera capture timestamp, in seconds (float64 in
the data, so millisecond ages survive). The data side already carries it:
`img_timestamp` is the sweep's capture time under honest geometry (stage 09
eval, stage 10 StaleSweepNuScenesDataset) and the camera's own keyframe time
otherwise (≈ ±40 ms intra-keyframe offsets).

The embedding MLP's last layer is zero-initialised, so at step 0 the model is
EXACTLY the pretrained one — unlike the stage-07 age port, which started from a
known-harmful configuration.

Registered in mmdet's HEADS; configs pull it in via custom_imports.
"""

import torch
import torch.nn as nn
from mmdet.models import HEADS

import projects.mmdet3d_plugin  # noqa: F401  (registers StreamPETRHead)
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from projects.mmdet3d_plugin.models.utils.misc import topk_gather

AGE_SCALE_S = 0.25            # 250 ms -> 1.0 at the MLP input
AGE_CLAMP_S = (-0.1, 1.0)     # small negatives: cameras captured after the lidar stamp


def camera_ages(data, batch):
    """Per-camera image age in seconds.

    Args:
        data: head input dict with 'timestamp' ([B] or [B,1]) and
            'img_timestamp' ([B, N]), both float64 seconds.
        batch: B.

    Returns:
        (B, N) float32 tensor, clamped to AGE_CLAMP_S.
    """
    ts = data["timestamp"].reshape(batch, -1)[:, :1].double()
    img_ts = data["img_timestamp"].reshape(batch, -1).double()
    return (ts - img_ts).float().clamp(*AGE_CLAMP_S)


@HEADS.register_module()
class AgeAwareStreamPETRHead(StreamPETRHead):
    """StreamPETRHead + zero-initialised per-camera age embedding on image-token positions."""

    def __init__(self, *args, age_hidden=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.age_embed = nn.Sequential(
            nn.Linear(1, age_hidden), nn.ReLU(inplace=True), nn.Linear(age_hidden, self.embed_dims))
        self._zero_age_output()

    def _zero_age_output(self):
        nn.init.zeros_(self.age_embed[-1].weight)
        nn.init.zeros_(self.age_embed[-1].bias)

    def init_weights(self):
        # also zeroed in __init__, since eval scripts build + load without init_weights()
        super().init_weights()
        self._zero_age_output()

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        """Parent 3D position embedding + the age embedding of each token's camera."""
        BN, H, W, _ = memory_centers.shape
        B = data["intrinsics"].size(0)
        N = BN // B
        pos_embed, cone = super().position_embeding(data, memory_centers, topk_indexes, img_metas)

        ages = camera_ages(data, B)                                        # (B, N)
        assert ages.shape[1] == N, (ages.shape, N)
        age_emb = self.age_embed((ages / AGE_SCALE_S).unsqueeze(-1).to(pos_embed.dtype))  # (B, N, C)
        cam_of_token = torch.arange(N, device=pos_embed.device).repeat_interleave(H * W)  # (N*H*W,)
        tok_emb = age_emb[:, cam_of_token]                                 # (B, N*H*W, C)
        if topk_indexes is not None:
            tok_emb = topk_gather(tok_emb, topk_indexes)
        return pos_embed + tok_emb, cone
