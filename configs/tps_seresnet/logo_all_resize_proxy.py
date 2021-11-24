_base_ = [
    '../_base_/models/tps_seresnet50_proxy.py',
    '../_base_/datasets/logo_all_resize.py',
    '../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=1866))

data = dict(
    samples_per_gpu=96,
    workers_per_gpu=6,
)
evaluation = dict(interval=1, metric=['accuracy', 'crossentropy'])
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 90, 95])
runner = dict(type='EpochBasedRunner', max_epochs=100)
