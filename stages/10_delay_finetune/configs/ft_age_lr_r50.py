"""
Stage-10b lr sweep: ft_age_r50.py with a larger learning rate on `age_embed` only.

Stage 10b's age input was ignored (indistinguishable from delay-FT in all 8
conditions). Most mundane explanation: the zero-initialised `age_embed` shares
lr 2e-5 with the pretrained neck/heads, and under AdamW it can move at most
~0.1 per weight over 7380 cosine iters (its last layer reached 0.02), while the
pretrained layers absorb the delay by hedging. This config multiplies the lr of
`age_embed` only; everything else (data, delay augmentation, schedule, init,
seed) is identical to ft_age_r50.py.

Env read at config-parse time: AGE_LR_MULT (required), plus the base's P_STALE,
RUN_NAME, WANDB_RUN_ID. Run with ARM=delay and RUN_NAME=ft_age_delay_p0.5_lrx<M>.
"""
import os

_base_ = ['./ft_age_r50.py']

AGE_LR_MULT = float(os.environ['AGE_LR_MULT'])

# Merged into the base custom_keys (img_backbone lr_mult 0.25 is kept).
optimizer = dict(paramwise_cfg=dict(custom_keys={'age_embed': dict(lr_mult=AGE_LR_MULT)}))
