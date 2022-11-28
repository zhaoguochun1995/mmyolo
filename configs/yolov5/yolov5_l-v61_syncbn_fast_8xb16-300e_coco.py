_base_ = './yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py'

deepen_factor = 1.0
widen_factor = 1.0

train_batch_size_per_gpu=8

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu),
    )

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu
    )

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
