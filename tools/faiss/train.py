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
        '/home/ubuntu/checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
    ),
    mobilenetv2=(
        'mmclassification/configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py',
        '/home/ubuntu/checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
    ),
    mobilenetv3=(
        'mmclassification/configs/mobilenet_v3/mobilenet_v3_small_imagenet.py',
        '/home/ubuntu/checkpoints/mobilenet_v3_small-047dcff4.pth'
    )
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a faiss index with image features.')
    parser.add_argument('dataset_config', help='test config file path')
    parser.add_argument(
        '--model',
        help='Specify the config to be used for feature extraction',
        default=(
            '/home/ubuntu/checkpoints/'
            'resnet50_batch256_imagenet_20200708-cfb998bf.pth'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        help='Specify the checkpoint to be used for feature extraction',
        default=(
            'mmclassification/configs/resnet/resnet50_b32x8_imagenet.py'
        ),
    )
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')

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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument(
        '--max-train',
        default=200,
        type=int,
        help='Maximum samples for training the index')
    parser.add_argument(
        '--batch-predict',
        default=100,
        type=int,
        help='Predict batch size')

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


def batch(iterable, n = 1):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def dataset_iterator(dataset):
    for i, samples in enumerate(dataset):
        img = samples['img']
        label = samples['gt_label']
        yield img, label


def feature_iterator(model, dataset, batch_size=700):
    for img_labels in batch(dataset_iterator(dataset), batch_size):
        imgs, labels = zip(*img_labels)
        with torch.no_grad():
            feats = model.extract_feat(
                torch.as_tensor(imgs)
            )
        yield feats[0].numpy().astype('float32'), np.array(labels)


def main():
    args = parse_args()

    cfg = retrieve_data_cfg(
        args.dataset_config,
        ['DefaultFormatBundle', 'Normalize', 'Collect'],
        args.cfg_options
    )

    model_cfg = retrieve_data_cfg(
        args.model,
        ['DefaultFormatBundle', 'Normalize', 'Collect'],
        args.cfg_options
    )

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **model_cfg.dist_params)

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

    add_gt_label_to_pipeline(dataset_train)
    add_gt_label_to_pipeline(dataset_test)

    import pickle
    with open('class_labels', 'wb+') as f:
        pickle.dump(class_dict, f)

    from mmcls.apis import init_model as feat_init_model
    model = feat_init_model(args.model, args.checkpoint, 'cpu')

    import faiss

    out_feats = model.head.in_channels

    for q_cls, q_args, q_kwargs, i_cls, i_args, i_kwargs in (
        (
                None, [], dict(),
                faiss.IndexFlatL2, [out_feats], dict(),
        ),
        (
                None, [], dict(),
                faiss.IndexFlatIP, [out_feats], dict(),
        ),
        (
                faiss.IndexFlatL2, [out_feats], dict(),
                faiss.IndexIVFFlat, [out_feats, 2, faiss.METRIC_L2], dict(),
        ),
    ):

        if q_cls:
            quantizer = q_cls(*q_args, **q_kwargs)
            i_args = (quantizer, *i_args)
        index = i_cls(*i_args, **i_kwargs)

        if q_cls:
            logging.info("Using index %s with quantizer %s", i_cls.__name__, q_cls.__name__)
        else:
            logging.info("Using index %s", i_cls.__name__)

        if hasattr(index, 'add_with_ids'):
            v = np.random.rand(1, out_feats).astype('float32')
            try:
                index.add_with_ids(v, np.array([1001]))
            except BaseException as e:
                msg = 'add_with_ids not implemented for this type of index'
                if msg in str(e):
                    logging.warning(
                        "Cannot use add_with_ids with %s", i_cls.__name__)
                    index = faiss.IndexIDMap(index)
                    index.add_with_ids(v, np.array([1001]))
                    continue

        logging.info("Extracting features for train batch #1...")
        feat_iter = feature_iterator(
            model, dataset_train, batch_size=args.max_train or 200)
        feats, labels = next(feat_iter)
        logging.debug("Obtained first batch to train (%s feats with len %s)", len(feats), len(feats[0]))
        if hasattr(index, 'train') and callable(index.train):
            logging.info("Train index using train batch #1")
            try:
                index.train(feats)
            except BaseException as be:
                if 'add_with_ids not implemented for this type of index' in str(be):
                    logging.error("Index with ids is not implemented for this index")
                    continue
                raise be

        logging.info(
            "Index %s samples of batch #1", feats.shape[0])
        index.add_with_ids(feats, labels)

        for i, (feats, labels) in enumerate(feat_iter, start=2):
            logging.info("Index %s samples of batch #%s", feats.shape[0], i)
            index.add_with_ids(feats, labels)

        logging.info("Extracting test features...")
        top_1_hit = 0
        top_5_hit = 0

        for i, (feats, labels) in enumerate(
                feature_iterator(
                    model,
                    dataset_test,
                    batch_size=args.batch_predict or 100),
                start=1):
            logging.info(
                "Search batch #%s (contains %s feats of length %s)",
                i, len(feats), len(feats[0]))
            ds, ids = index.search(feats, 5)
            sorted_by_distance = list(
                list(
                    map(lambda x: x[1],
                        sorted(zip(ind_ds, ind_ids), key=lambda x: x[0])
                        )
                ) for ind_ds, ind_ids in zip(ds, ids))

            for true_label, candidates in zip(labels, sorted_by_distance):
                if true_label == candidates[0]:
                    top_1_hit += 1
                    top_5_hit += 1
                elif true_label in candidates:
                    top_5_hit += 1

        logging.info(f"Top-1 Accuracy: {top_1_hit / len(labels)}")
        logging.info(f"Top-5 Accuracy: {top_5_hit / len(labels)}")

        # faiss.write_index(index, f'index_100_{args.feature_extractor}')


if __name__ == '__main__':
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    main()
