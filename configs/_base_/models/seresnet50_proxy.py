# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ProxyLinearClsHead',
        num_classes=1000,
        in_channels=2048,
        out_features=512
    ))
