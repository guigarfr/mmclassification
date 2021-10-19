model = dict(
    type='ImageClassifier',
    pretrained='/home/ubuntu/checkpoints/mobilenet_v3_small-047dcff4.pth',
    backbone=dict(type='MobileNetV3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=3793,
        in_channels=576,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
data_root = '/home/ubuntu/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
all_classes = '/home/ubuntu/data/all_classes.txt'
ob_root = '/home/ubuntu/data/OpenBrands/'
train_openbrand = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_1_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_1/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_2_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_2/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_3_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_3/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_4_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_4/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_5_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_5/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_6_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_6/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_7_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_7/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_8_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_8/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_9_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_9/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_10_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_10/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_11_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_11/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_12_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_12/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_13_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_13/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_14_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_14/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_15_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_15/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_16_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_16/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_17_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_17/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_18_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_18/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_19_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_19/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/train_20210409_20_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_20/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ])
    ])
validation_openbrand = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_1_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_1/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_2_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_2/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_3_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_3/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_4_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_4/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_5_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_5/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_6_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_6/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_7_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_7/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_8_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_8/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_9_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_9/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_10_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_10/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_11_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_11/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_12_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_12/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_13_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_13/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_14_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_14/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_15_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_15/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_16_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_16/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_17_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_17/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_18_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_18/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_19_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_19/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_20_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_20/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
test_openbrand = dict(
    type='ConcatDataset',
    datasets=[
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_1_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_1/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_2_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_2/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_3_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_3/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_4_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_4/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_5_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_5/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_6_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_6/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_7_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_7/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_8_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_8/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_9_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_9/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_10_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_10/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_11_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_11/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_12_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_12/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_13_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_13/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_14_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_14/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_15_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_15/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_16_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_16/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_17_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_17/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_18_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_18/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_19_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_19/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='OpenBrandDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/OpenBrands/annotations/test_20210409_20_reduced.json',
            data_prefix=
            '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_20/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
ld_root = '/home/ubuntu/data/LogoDet-3K/'
rp_root = '/home/ubuntu/data/logo_dataset/'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=[
        dict(
            type='XMLDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file='/home/ubuntu/data/LogoDet-3K/train_reduced.txt',
            ann_subdir='',
            data_prefix='/home/ubuntu/data/LogoDet-3K/',
            img_subdir='',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='XMLDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file='/home/ubuntu/data/logo_dataset/ImageSets/Main/train.txt',
            ann_subdir='Annotations',
            data_prefix='/home/ubuntu/data/logo_dataset/',
            img_subdir='JPEGImages',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img', 'gt_label'])
            ]),
        dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_1_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_1/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_2_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_2/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_3_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_3/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_4_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_4/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_5_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_5/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_6_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_6/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_7_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_7/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_8_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_8/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_9_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_9/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_10_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_10/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_11_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_11/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_12_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_12/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_13_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_13/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_14_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_14/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_15_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_15/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_16_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_16/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_17_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_17/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_18_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_18/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_19_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_19/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/train_20210409_20_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_20/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='RandomFlip',
                            flip_prob=0.5,
                            direction='horizontal'),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img', 'gt_label'])
                    ])
            ])
    ],
    val=[
        dict(
            type='XMLDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file='/home/ubuntu/data/LogoDet-3K/val_reduced.txt',
            ann_subdir='',
            data_prefix='/home/ubuntu/data/LogoDet-3K/',
            img_subdir='',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='XMLDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file=
            '/home/ubuntu/data/logo_dataset/ImageSets/Main/validation.txt',
            ann_subdir='Annotations',
            data_prefix='/home/ubuntu/data/logo_dataset/',
            img_subdir='JPEGImages',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_1_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_1/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_2_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_2/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_3_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_3/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_4_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_4/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_5_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_5/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_6_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_6/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_7_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_7/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_8_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_8/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_9_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_9/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_10_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_10/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_11_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_11/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_12_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_12/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_13_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_13/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_14_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_14/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_15_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_15/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_16_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_16/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_17_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_17/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_18_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_18/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_19_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_19/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/validation_20210409_20_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_20/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ])
    ],
    test=[
        dict(
            type='XMLDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file='/home/ubuntu/data/LogoDet-3K/test_reduced.txt',
            ann_subdir='',
            data_prefix='/home/ubuntu/data/LogoDet-3K/',
            img_subdir='',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='XMLDataset',
            classes='/home/ubuntu/data/all_classes.txt',
            ann_file='/home/ubuntu/data/logo_dataset/ImageSets/Main/test.txt',
            ann_subdir='Annotations',
            data_prefix='/home/ubuntu/data/logo_dataset/',
            img_subdir='JPEGImages',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='CropBoundingBox'),
                dict(type='Resize', size=(224, 224)),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ]),
        dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_1_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_1/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_2_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_2/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_3_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_3/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_4_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_4/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_5_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_5/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_6_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_6/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_7_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_7/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_8_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_8/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_9_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_9/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_10_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_10/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_11_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_11/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_12_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_12/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_13_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_13/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_14_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_14/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_15_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_15/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_16_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_16/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_17_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_17/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_18_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_18/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_19_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_19/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]),
                dict(
                    type='OpenBrandDataset',
                    classes='/home/ubuntu/data/all_classes.txt',
                    ann_file=
                    '/home/ubuntu/data/OpenBrands/annotations/test_20210409_20_reduced.json',
                    data_prefix=
                    '/home/ubuntu/data/OpenBrands/电商标识检测大赛_train_20210409_20/',
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='CropBoundingBox'),
                        dict(type='Resize', size=(224, 224)),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/home/ubuntu/train_checkpoints/mobilenet_ob/latest.pth'
workflow = [('train', 1)]
evaluation = dict(interval=5, metric=['accuracy', 'crossentropy'])
optimizer = dict(
    type='RMSprop',
    lr=0.008,
    alpha=0.9,
    momentum=0.9,
    eps=0.0316,
    weight_decay=1e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=2, gamma=0.973, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=600)
work_dir = '/home/ubuntu/train_checkpoints/mobilenetv3_all'
gpu_ids = range(0, 1)
