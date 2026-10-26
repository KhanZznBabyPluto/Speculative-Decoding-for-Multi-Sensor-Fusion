"""
Stage-10 delay fine-tune + per-camera age conditioning (AgeAwareStreamPETRHead).

Identical to ft_stale_r50.py (same data, delay augmentation, frozen backbone,
lr, schedule, init) except the 3D head receives each camera's image age as a
zero-initialised embedding on its image-token positions. Run with ARM=delay and
RUN_NAME=ft_age_delay_p0.5 so it gets its own work dir and W&B run.
"""
_base_ = ['./ft_stale_r50.py']

custom_imports = dict(imports=['stale_dataset', 'age_head'], allow_failed_imports=False)
model = dict(pts_bbox_head=dict(type='AgeAwareStreamPETRHead', age_hidden=128))
