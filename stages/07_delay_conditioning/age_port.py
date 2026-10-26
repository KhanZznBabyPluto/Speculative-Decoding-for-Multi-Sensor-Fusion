"""A trainable age-conditioning port for StreamPETR's image-born queries.

THE PROBLEM THIS TESTS
----------------------
StreamPETR conditions its *carried memory* tokens on real motion information —
`cat([memory_velo(2), memory_timestamp(1), memory_egopose(12)])` -> 15 dims ->
`nerf_positional_encoding` -> 180 dims -> two shared `MLN` (FiLM) modules. Its
*newly-born image queries* go through the same modules with that 15-vector
hardcoded to `(0, 0, 0, identity)`: "you are here, now"
([streampetr_head.py:426,438](../../StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py#L426)).

In fresh operation that is correct by definition, so nothing is broken upstream.
Under staleness it is a lie, and the 2026-07-17/18 audit showed that simply
telling the truth makes things *worse*: feeding the true `(T, dt)` through those
shared modules on top of otherwise-correct geometry costs
-0.032 mAP (95% CI [-0.041, -0.023], 1/23 scenes, p=0.000).

The natural explanation is distributional, not architectural: the modules were
never trained on the pair (image-born query, non-zero motion code). Every
training step fed them exactly `(0, 0, 0, I)` on that path. So the question this
module exists to answer is:

    Is the conditioning chimera a property of the representation,
    or just an untrained input port?

WHAT THIS ADDS
--------------
A dedicated pair of `MLN` modules for the query path, plus a dedicated time
embedding, initialised as exact copies of the frozen shared ones. Two properties
follow:

  * **Exact-identity init.** With `dt=0` and `T=I` the copies are numerically
    identical to the originals, so a fresh-frame forward is bit-identical to the
    pretrained model. Training can only move behaviour where delay > 0. This is
    the same zero-init-residual discipline the LTC used, for the same reason: we
    never start from a regime worse than the baseline.
  * **The memory path keeps working.** Fine-tuning the *shared* modules in place
    would perturb carried-memory conditioning, which is currently correct and is
    doing real work. Giving the query path its own port isolates the change.

~350K trainable parameters against a ~60M frozen model.

WHAT WOULD FALSIFY THE "UNTRAINED PORT" HYPOTHESIS
--------------------------------------------------
Fine-tuning this port under delay augmentation and evaluating at delay=1:

    mAP > 0.1849 (the honest key-side baseline)  -> the chimera dissolves with
        training; it was distribution shift, i.e. an engineering problem, and
        age-conditioned input contracts are a constructive fix.
    mAP <= 0.1849                                -> the chimera survives even
        with the port explicitly trained open; the claim that these latents
        resist external conditioning becomes structural rather than asserted.

Both outcomes are informative, which is why this is worth the GPU time.
"""

import copy

import torch
import torch.nn as nn

NUSC_DT = 0.5  # keyframe spacing (2 Hz), seconds


class AgeConditioningPort(nn.Module):
    """Dedicated FiLM conditioning for image-born queries, keyed on observation age.

    Args:
        head: the live `StreamPETRHead`; its `ego_pose_pe`, `ego_pose_memory` and
            `time_embedding` are deep-copied to initialise this port, and are
            themselves left frozen and untouched.
    """

    def __init__(self, head):
        super().__init__()
        # Deep copies, so training these never perturbs the memory path.
        self.age_pose_pe = copy.deepcopy(head.ego_pose_pe)
        self.age_pose_memory = copy.deepcopy(head.ego_pose_memory)
        self.age_time_embedding = copy.deepcopy(head.time_embedding)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def install_age_port(model, port):
    """Route the head's query path through `port`, keeping everything else frozen.

    Replaces `StreamPETRHead.temporal_alignment` with a version identical to
    upstream except that image-born queries are conditioned on the true
    `(T, dt)` through `port` instead of on `(identity, 0)` through the shared
    modules. The carried-memory branch is byte-for-byte the upstream code.

    The `(T, dt)` for the current frame is read from `model._age_state`, which
    the training/eval driver sets each step (see `train_age_port.py`). When that
    is absent or zero-delay, the port receives exactly the inputs the frozen
    modules would have seen, so behaviour matches the pretrained model.

    Returns the original bound method, so the caller can restore it.
    """
    from projects.mmdet3d_plugin.models.utils.positional_encoding import (
        nerf_positional_encoding, pos2posemb1d, pos2posemb3d)

    head = model.pts_bbox_head
    original = head.temporal_alignment

    def temporal_alignment(query_pos, tgt, reference_points):
        B = query_pos.size(0)
        state = getattr(model, "_age_state", None)
        T = None if state is None else state.get("T")
        dt = 0.0 if state is None else float(state.get("dt", 0.0))

        temp_reference_point = ((head.memory_reference_point - head.pc_range[:3])
                                / (head.pc_range[3:6] - head.pc_range[0:3]))
        temp_pos = head.query_embedding(pos2posemb3d(temp_reference_point))
        temp_memory = head.memory_embedding

        if T is None:
            rec_ego_pose = torch.eye(4, device=query_pos.device)\
                .unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        else:
            rec_ego_pose = T.to(query_pos.device).view(1, 1, 4, 4)\
                .repeat(B, query_pos.size(1), 1, 1).float()

        if head.with_ego_pos:
            # Slots [0:3] mirror the memory layout [velo_x, velo_y, timestamp];
            # upstream leaves all three at zero for image-born queries.
            rec_prefix = torch.zeros_like(reference_points[..., :3])
            rec_prefix[..., 2] = dt
            rec_ego_motion = torch.cat(
                [rec_prefix, rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)

            # THE ONE CHANGE: the query path uses the trainable port.
            tgt = port.age_pose_memory(tgt, rec_ego_motion)
            query_pos = port.age_pose_pe(query_pos, rec_ego_motion)

            # Carried memory keeps the frozen shared modules and real values.
            memory_ego_motion = torch.cat(
                [head.memory_velo, head.memory_timestamp,
                 head.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = head.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = head.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos = query_pos + port.age_time_embedding(
            pos2posemb1d(torch.full_like(reference_points[..., :1], dt)))
        temp_pos = temp_pos + head.time_embedding(
            pos2posemb1d(head.memory_timestamp).float())

        if head.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :head.num_propagated]], dim=1)
            query_pos = torch.cat(
                [query_pos, temp_pos[:, :head.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, :head.num_propagated]],
                dim=1)
            n_out = query_pos.shape[1] + head.num_propagated
            if T is None:
                rec_ego_pose = torch.eye(4, device=query_pos.device)\
                    .unsqueeze(0).unsqueeze(0).repeat(B, n_out, 1, 1)
            else:
                rec_ego_pose = T.to(query_pos.device).view(1, 1, 4, 4)\
                    .repeat(B, n_out, 1, 1).float()
            temp_memory = temp_memory[:, head.num_propagated:]
            temp_pos = temp_pos[:, head.num_propagated:]

        return (tgt, query_pos, reference_points, temp_memory, temp_pos,
                rec_ego_pose)

    head.temporal_alignment = temporal_alignment
    return original


def freeze_all_but_port(model, port):
    """Freeze every model parameter; leave only the port trainable.

    Returns (n_frozen, n_trainable) for logging — a sanity check that belongs in
    the log of every run, because silently training the backbone on 62 scenes
    would be catastrophic and invisible in the loss curve.
    """
    n_frozen = 0
    for p in model.parameters():
        p.requires_grad_(False)
        n_frozen += p.numel()
    for p in port.parameters():
        p.requires_grad_(True)
    return n_frozen, port.n_params()
