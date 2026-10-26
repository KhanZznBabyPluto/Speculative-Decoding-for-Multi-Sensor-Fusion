"""
Fine-tune StreamPETR R50 (704x256, seq) with random per-camera sub-keyframe delay.

Inherits the exact config every earlier measurement used and changes only:
  * train data: the 62 local train scenes, via StaleSweepNuScenesDataset
    (P_STALE env: per-camera probability of a 1-3 sweep delay; 0 = control arm)
  * image backbone frozen (stem + all 4 stages); neck + heads train
  * init from the published checkpoint, lr 2e-5, 12 epochs on 2 GPUs x bs 2
  * checkpoints every half epoch (kept, for mAP-based selection later)
  * W&B: khangeldinansar-kaust / multisensor-staleness; run id from env so a
    resubmitted chain continues the same run

Env read at config-parse time: P_STALE (default 0.5), RUN_NAME, WANDB_RUN_ID.
"""
import os

_base_ = ['/ibex/user/khangea/multisensor/StreamPETR/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py']

PROJECT_DIR = '/ibex/user/khangea/multisensor'
P_STALE = float(os.environ.get('P_STALE', '0.5'))
RUN_NAME = os.environ.get('RUN_NAME', f'ft_stale_p{P_STALE}')
WANDB_RUN_ID = os.environ.get('WANDB_RUN_ID', RUN_NAME)

custom_imports = dict(imports=['stale_dataset'], allow_failed_imports=False)

num_gpus = 2
batch_size = 2
num_train_samples = 2462                     # 62 scenes in data_withtrain
num_iters_per_epoch = num_train_samples // (num_gpus * batch_size)   # 615
num_epochs = 12
max_iters = num_epochs * num_iters_per_epoch                          # 7380

model = dict(
    img_backbone=dict(pretrained=None, frozen_stages=4),   # weights come from load_from
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='StaleSweepNuScenesDataset',
        ann_file=PROJECT_DIR + '/data_withtrain/nuscenes2d_temporal_infos_train.pkl',
        sweep_index=PROJECT_DIR + '/stages/09_subframe_async/dumps/sweep_index_train.pkl',
        labels2d=PROJECT_DIR + '/stages/10_delay_finetune/dumps/sweep_labels2d_train.pkl',
        p_stale=P_STALE,
        max_j=3,
    ),
)

optimizer = dict(lr=2e-5)
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=200,
                 warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)

runner = dict(type='IterBasedRunner', max_iters=max_iters)
checkpoint_config = dict(interval=num_iters_per_epoch // 2, max_keep_ckpts=30)
evaluation = dict(interval=10 ** 9)   # eval is done separately with stages/09 eval_async_cams.py

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook', by_epoch=False, log_artifact=False,
             init_kwargs=dict(entity='khangeldinansar-kaust', project='multisensor-staleness',
                              name=RUN_NAME, id=WANDB_RUN_ID, resume='allow',
                              group='stage10_delay_finetune',
                              config=dict(p_stale=P_STALE, max_j=3, lr=2e-5,
                                          frozen_backbone=True, max_iters=max_iters,
                                          train_scenes=62))),
    ])

load_from = PROJECT_DIR + '/checkpoints/stream_petr_r50_flash_704_bs2_seq_90e.pth'
resume_from = None
