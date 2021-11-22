# model settings
model = dict(
    type='ImageClassifier',
    pretrained='/home/ubuntu/checkpoints/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ProxyLinearClsHead',
        num_classes=0,
        in_channels=2048,
        out_features=512,
    ))
