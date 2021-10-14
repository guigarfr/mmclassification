# model settings
num_classes = 515

model = dict(
    type='ImageClassifier',
    pretrained='/home/ubuntu/checkpoints/mobilenet_v3_small-047dcff4.pth',
    backbone=dict(type='MobileNetV3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=num_classes,
        in_channels=576,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
