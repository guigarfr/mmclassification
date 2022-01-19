# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        ),
        frozen_stages=4,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ProxyLinearClsHead',
        num_classes=0,
        in_channels=2048,
        out_features=2048,
    )
)
