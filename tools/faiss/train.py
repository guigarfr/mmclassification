# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import warnings

from collections.abc import Sequence
import numpy as np
import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmcls.datasets import (build_dataloader, build_dataset, ConcatDataset)


feature_extractors = dict(
    resnet50=(
        'mmclassification/configs/resnet/resnet50_b32x8_imagenet.py',
        'mmclassification/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
    ),
    mobilenetv2=(
        'mmclassification/configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py',
        'mmclassification/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
    ),
    mobilenetv3=(
        'configs/mobilenet_v3/mobilenet_v3_small_imagenet.py',
        '/home/ubuntu/checkpoints/mobilenet_v3_small-047dcff4.pth'
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--feature-extractor',
        help='Specify the feature extractor to be used',
        choices=list(feature_extractors.keys()),
        default='resnet50'
    )
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


import itertools

def get_dataset_config(cfg):
    if isinstance(cfg, Sequence):
        return itertools.chain.from_iterable([get_dataset_config(c) for c in
                                              cfg])
    elif isinstance(cfg, dict) and \
            hasattr(cfg, 'datasets') and \
            isinstance(cfg.datasets, list):
        return itertools.chain.from_iterable([get_dataset_config(c) for c in
                                              cfg['datasets']])
    else:
        return [cfg]


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
        'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    for c in get_dataset_config(train_data_cfg):
        skip_pipeline_steps(c)

    return cfg


def add_gt_label_to_pipeline(dataset):
    if isinstance(dataset, ConcatDataset):
        for ind_dataset in dataset.datasets:
            add_gt_label_to_pipeline(ind_dataset)
    else:
        dataset.pipeline.transforms[-1].keys.append('gt_label')


def compute_features(model, dataset):
    train = []
    for i, samples in enumerate(dataset):
        img = samples['img']
        label = samples['gt_label']

        with torch.no_grad():
            feats = model.extract_feat(
                torch.as_tensor([img])
            )[0]
            train.append((feats.numpy().astype('float32'), label))

    return train


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = retrieve_data_cfg(
        args.config,
        ['DefaultFormatBundle', 'Normalize', 'Collect'],
        args.cfg_options
    )

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset_train = build_dataset(cfg.data.val)
    dataset_test = build_dataset(cfg.data.test)

    if isinstance(dataset_train, ConcatDataset):
        class_dict = {}
        for d in dataset_train.datasets:
            if isinstance(d, ConcatDataset):
                for ds in d.datasets:
                    class_dict.update(ds.cat2label)
            else:
                class_dict.update(d.cat2label)
    else:
        class_dict = dataset_train.cat2label

    import pickle
    with open('class_labels', 'wb+') as f:
        pickle.dump(class_dict, f)
    # Get train samples

    config_path, checkpoint_path = feature_extractors[args.feature_extractor]
    feature_cfg = mmcv.Config.fromfile(config_path)
    from mmcls.apis import init_model as feat_init_model

    feature_model = feat_init_model(feature_cfg, checkpoint_path, 'cpu')
    add_gt_label_to_pipeline(dataset_train)
    add_gt_label_to_pipeline(dataset_test)

    import faiss

    out_feats = feature_cfg._cfg_dict['model']['head']['in_channels']
    quantizer = faiss.IndexFlatL2(out_feats)
    index = faiss.IndexIVFFlat(quantizer, out_feats, 2, faiss.METRIC_L2)

    if not hasattr(index, 'add_with_ids') and callable(index.add_with_ids):
        index = faiss.IndexIDMap2(index)

    print("Training...")
    print("Extracting features...")
    train = compute_features(feature_model, dataset_train)
    feats, labels = zip(*train)
    print(len(feats))
    feats = np.vstack(feats)
    print(feats.shape)
    print(feats[0].shape)
    print("Faiss Train + Index")
    if hasattr(index, 'train') and callable(index.train):
        index.train(feats)
    index.add_with_ids(feats, np.array(labels).flatten())

    print("Testing...")
    print("Extracting features...")
    test = compute_features(feature_model, dataset_test)
    feats, labels = zip(*test)
    print(len(feats))
    feats = np.vstack(feats)
    print(feats.shape)
    print(feats[0].shape)
    print("Faiss Testing")
    ds, ids = index.search(feats, 5)
    sorted_by_distance = list(
        list(
            map(lambda x: x[1],
                sorted(zip(ind_ds, ind_ids), key=lambda x: x[0])
                )
        ) for ind_ds, ind_ids in zip(ds, ids))

    top_1_hit = 0
    top_5_hit = 0
    for true_label, candidates in zip(labels, sorted_by_distance):
        if true_label == candidates[0]:
            top_1_hit += 1
            top_5_hit += 1
        elif true_label in candidates:
            top_5_hit += 1

    print(f"Top-1 Accuracy: {top_1_hit / len(labels)}")
    print(f"Top-5 Accuracy: {top_5_hit / len(labels)}")

    # faiss.write_index(index, f'index_100_{args.feature_extractor}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
