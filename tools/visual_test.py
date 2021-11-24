# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import random
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import matplotlib.pyplot as plt

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
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
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def convert_label(label_idx, dataset):
    return dataset.CLASSES[label_idx]

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    pipeline = cfg.data.test[0].pipeline
    # Assuming that pipeline[-1] is Collect
    pipeline[-1]['keys'].append('gt_label')
    for dataset in cfg.data.test:
        dataset.pipeline = pipeline

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    samples_idx = random.sample(range(len(dataset)), 16)
    batch_imgs = [dataset[idx]['img'] for idx in samples_idx]
    batch_labels = [dataset[idx]['gt_label'] for idx in samples_idx]

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.CLASSES = CLASSES
    with torch.no_grad():
        result = model.simple_test(torch.stack(batch_imgs), {})

    for img, true_label, pred_label in zip(batch_imgs, batch_labels, result):
        img_array = np.rollaxis(img.numpy(), 0, 3)
        img_min, img_max = np.min(img_array), np.max(img_array)
        img_array = (img_array - img_min) / (img_max - img_min)
        plt.imshow(img_array)
        plt.show()
        print()
        print(f"True label: {convert_label(true_label, dataset)}")
        print(
            f"Predicted label: {convert_label(np.argmax(pred_label), dataset)}")
        print(f"With confidence: {np.max(pred_label)}")


if __name__ == '__main__':
    main()
