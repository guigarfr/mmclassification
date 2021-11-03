# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections.abc import Sequence
from collections import Counter
import itertools
import json
import numpy as np

import matplotlib.pyplot as plt
import mmcv
from mmcv import Config, DictAction

from mmcls.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
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
    parser.add_argument('--just-classes', action='store_true')
    parser.add_argument('-subset', choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()
    return args


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


def patch_config_pipeline(dataset_config, pipeline):
    if isinstance(dataset_config, list):
        for sub_dataset in dataset_config:
            patch_config_pipeline(sub_dataset, pipeline)
    elif isinstance(dataset_config, dict) and dataset_config.type == 'ConcatDataset':
        for sub_dataset in dataset_config.datasets:
            patch_config_pipeline(sub_dataset, pipeline)
    else:
        dataset_config.pipeline = pipeline


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    if args.subset == 'train':
        config_dataset = cfg.data.train
    elif args.subset == 'val':
        config_dataset = cfg.data.val
    elif args.subset == 'test':
        config_dataset = cfg.data.test
    else:
        raise Exception("Subset no correctly specified.")

    if args.just_classes:
        just_classes_pipeline = [
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Collect', keys=['gt_labels']),
        ]
        patch_config_pipeline(config_dataset, just_classes_pipeline)

    dataset = build_dataset(config_dataset)

    progress_bar = mmcv.ProgressBar(len(dataset))

    classes = [0] * 3579
    for item in dataset:
        if args.just_classes:
            if 'gt_labels' in item:
                classes[int(item['gt_labels'][0][0])] += 1
        else:
            image = np.rollaxis(item['img'].detach().numpy(), 0, 3)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            plt.imshow(image)
            plt.show()

        progress_bar.update()

    if classes:
        with open('compute_class_distribution.json', 'w') as tfile:
            json.dump(classes, tfile)


if __name__ == '__main__':
    main()
