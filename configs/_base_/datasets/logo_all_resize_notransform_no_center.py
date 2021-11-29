data_root = '/home/ubuntu/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

policies = [
    [
        dict(type='Posterize', bits=4, prob=0.4),
        dict(type='Rotate', angle=30., prob=0.6, pad_val=0)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
    [
        dict(type='Posterize', bits=5, prob=0.6),
        dict(type='Posterize', bits=5, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8, pad_val=0)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 6, prob=0.6),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Posterize', bits=6, prob=0.8),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='Rotate', angle=10., prob=0.2, pad_val=0),
        dict(type='Solarize', thr=256 / 9, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.6),
        dict(type='Posterize', bits=5, prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8, pad_val=0),
        dict(type='ColorTransform', magnitude=0., prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30., prob=0.4, pad_val=0),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.0),
     dict(type='Equalize', prob=0.8)],
    # [
    #     dict(type='Invert', prob=0.6),
    #     dict(type='Equalize', prob=1.)
    # ],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8, pad_val=0),
        dict(type='ColorTransform', magnitude=0.2, prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0.8, prob=0.8),
        dict(type='Solarize', thr=256 / 9 * 2, prob=0.8)
    ],
    [
        dict(type='Sharpness', magnitude=0.7, prob=0.4),
        dict(type='Invert', prob=0.6)
    ],
    [
        dict(
            type='Shear',
            magnitude=0.3 / 9 * 5,
            prob=0.6,
            direction='horizontal'),
        dict(type='Equalize', prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    # [
    #     dict(type='Invert', prob=0.6),
    #     dict(type='Equalize', prob=1.)
    # ],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, -1), adaptive_side='long',
         only_resize_bigger=True),
    dict(type='Pad', size=(224, 224), centered=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CropBoundingBox'),
    dict(type='Resize', size=(224, -1), adaptive_side='long',
         only_resize_bigger=True),
    dict(type='Pad', size=(224, 224), centered=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

all_classes = data_root + 'all_classes_v4.txt'
class_unifier = data_root + 'class_unification_dict_v4.json'
ob_root = data_root + 'OpenBrands/'
data_folders = [
    '电商标识检测大赛_train_20210409_1/',
    '电商标识检测大赛_train_20210409_2/',
    '电商标识检测大赛_train_20210409_3/',
    '电商标识检测大赛_train_20210409_4/',
    '电商标识检测大赛_train_20210409_5/',
    '电商标识检测大赛_train_20210409_6/',
    '电商标识检测大赛_train_20210409_7/',
    '电商标识检测大赛_train_20210409_8/',
    '电商标识检测大赛_train_20210409_9/',
    '电商标识检测大赛_train_20210409_10/',
    '电商标识检测大赛_train_20210409_11/',
    '电商标识检测大赛_train_20210409_12/',
    '电商标识检测大赛_train_20210409_13/',
    '电商标识检测大赛_train_20210409_14/',
    '电商标识检测大赛_train_20210409_15/',
    '电商标识检测大赛_train_20210409_16/',
    '电商标识检测大赛_train_20210409_17/',
    '电商标识检测大赛_train_20210409_18/',
    '电商标识检测大赛_train_20210409_19/',
    '电商标识检测大赛_train_20210409_20/',
]
train_openbrand = dict(
    type='ConcatDatasetBuilder',
    dataset_type='OpenBrandDataset',
    classes=all_classes,
    pipeline=train_pipeline,
    ann_files=['train_20210409_1_reduced.json',
               'train_20210409_2_reduced.json',
               'train_20210409_3_reduced.json',
               'train_20210409_4_reduced.json',
               'train_20210409_5_reduced.json',
               'train_20210409_6_reduced.json',
               'train_20210409_7_reduced.json',
               'train_20210409_8_reduced.json',
               'train_20210409_9_reduced.json',
               'train_20210409_10_reduced.json',
               'train_20210409_11_reduced.json',
               'train_20210409_12_reduced.json',
               'train_20210409_13_reduced.json',
               'train_20210409_14_reduced.json',
               'train_20210409_15_reduced.json',
               'train_20210409_16_reduced.json',
               'train_20210409_17_reduced.json',
               'train_20210409_18_reduced.json',
               'train_20210409_19_reduced.json',
               'train_20210409_20_reduced.json',
               ],
    data_folders=data_folders,
    ann_prefix=ob_root+'annotations/',
    data_prefix=ob_root,
    class_unifier=class_unifier,
)

validation_openbrand = dict(
    type='ConcatDatasetBuilder',
    dataset_type='OpenBrandDataset',
    classes=all_classes,
    pipeline=test_pipeline,
    ann_files=[
        'validation_20210409_1_reduced.json',
        'validation_20210409_2_reduced.json',
        'validation_20210409_3_reduced.json',
        'validation_20210409_4_reduced.json',
        'validation_20210409_5_reduced.json',
        'validation_20210409_6_reduced.json',
        'validation_20210409_7_reduced.json',
        'validation_20210409_8_reduced.json',
        'validation_20210409_9_reduced.json',
        'validation_20210409_10_reduced.json',
        'validation_20210409_11_reduced.json',
        'validation_20210409_12_reduced.json',
        'validation_20210409_13_reduced.json',
        'validation_20210409_14_reduced.json',
        'validation_20210409_15_reduced.json',
        'validation_20210409_16_reduced.json',
        'validation_20210409_17_reduced.json',
        'validation_20210409_18_reduced.json',
        'validation_20210409_19_reduced.json',
        'validation_20210409_20_reduced.json',
    ],
    data_folders=data_folders,
    ann_prefix=ob_root+'annotations/',
    data_prefix=ob_root,
    class_unifier=class_unifier,
)

test_openbrand = dict(
    type='ConcatDatasetBuilder',
    dataset_type='OpenBrandDataset',
    classes=all_classes,
    pipeline=test_pipeline,
    ann_files=[
        'test_20210409_1_reduced.json',
        'test_20210409_2_reduced.json',
        'test_20210409_3_reduced.json',
        'test_20210409_4_reduced.json',
        'test_20210409_5_reduced.json',
        'test_20210409_6_reduced.json',
        'test_20210409_7_reduced.json',
        'test_20210409_8_reduced.json',
        'test_20210409_9_reduced.json',
        'test_20210409_10_reduced.json',
        'test_20210409_11_reduced.json',
        'test_20210409_12_reduced.json',
        'test_20210409_13_reduced.json',
        'test_20210409_14_reduced.json',
        'test_20210409_15_reduced.json',
        'test_20210409_16_reduced.json',
        'test_20210409_17_reduced.json',
        'test_20210409_18_reduced.json',
        'test_20210409_19_reduced.json',
        'test_20210409_20_reduced.json',
    ],
    data_folders=data_folders,
    ann_prefix=ob_root+'annotations/',
    data_prefix=ob_root,
    class_unifier=class_unifier,
)

ld_root = data_root + 'LogoDet-3K/'
rp_root = data_root + 'logo_dataset/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=[
            dict(
                type='XMLDataset',
                classes=all_classes,
                ann_file=ld_root + 'train_reduced.txt',
                ann_subdir='',
                data_prefix=ld_root,
                img_subdir='',
                pipeline=train_pipeline,
                class_unifier=class_unifier,
            ),
            dict(
                type='XMLDataset',
                classes=all_classes,
                ann_file=rp_root + 'ImageSets/Main/train.txt',
                ann_subdir='Annotations',
                data_prefix=rp_root,
                img_subdir='JPEGImages',
                pipeline=train_pipeline,
                class_unifier=class_unifier,
            ),
            train_openbrand,
        ],
    val=[
            dict(
                type='XMLDataset',
                classes=all_classes,
                ann_file=ld_root + 'val_reduced.txt',
                ann_subdir='',
                data_prefix=ld_root,
                img_subdir='',
                pipeline=test_pipeline,
                class_unifier=class_unifier,
            ),
            dict(
                type='XMLDataset',
                classes=all_classes,
                ann_file=rp_root + 'ImageSets/Main/validation.txt',
                ann_subdir='Annotations',
                data_prefix=rp_root,
                img_subdir='JPEGImages',
                pipeline=test_pipeline,
                class_unifier=class_unifier,
            ),
            validation_openbrand,
        ],
    test=[
             dict(
                 type='XMLDataset',
                 classes=all_classes,
                 ann_file=ld_root + 'test_reduced.txt',
                 ann_subdir='',
                 data_prefix=ld_root,
                 img_subdir='',
                 pipeline=test_pipeline,
                 class_unifier=class_unifier,
             ),
             dict(
                 type='XMLDataset',
                 classes=all_classes,
                 ann_file=rp_root + 'ImageSets/Main/test.txt',
                 ann_subdir='Annotations',
                 data_prefix=rp_root,
                 img_subdir='JPEGImages',
                 pipeline=test_pipeline,
                 class_unifier=class_unifier,
            ),
            test_openbrand,
        ]
)
