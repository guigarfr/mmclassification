# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import collections
import datetime
import logging
import os
import tempfile
import time
import warnings

from collections.abc import Sequence

import faiss
import numpy as np
import torch
import sys
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

import humanize

import psutil

from mmcls.datasets import (build_dataloader, build_dataset, ConcatDataset)


def get_memory(index):
    fn = tempfile.mkstemp('.index', prefix='faiss_')[1]
    faiss.write_index(index, fn)
    return os.path.getsize(fn)

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
        '--k',
        type=int,
        default=5,
        help='search k top similar results')

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

    parser.add_argument(
        '--out',
        default=None,
        type=argparse.FileType("w+"),
        help='File to store output.')

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


def batch(iterable, n=1):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def dataset_iterator(dataset, max_samples=0):
    for i, samples in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        img = samples['img']
        label = samples['gt_label']
        yield img, label


def feature_iterator(model, dataset, batch_size=700):
    my_iter = dataset_iterator(dataset)
    if batch_size:
        my_iter = batch(my_iter, batch_size)

    for img_labels in my_iter:
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

    if args.out is not None:
        args.out.write(
            'Index\tMetric\t'
            'Sec-Train\tSec-Index\tSec-Search\t'
            'Bytes-faiss\tBytes-idx-ram\tBytes-idx-disk'
            '{}\n'.format(
                '\t'.join(['Top-{}-Acc'.format(k) for k in range(1, args.k + 1)])
            )

        )

    import faiss

    out_feats = model.head.in_channels

    base_faiss_mem = faiss.get_mem_usage_kb() * 1024

    index = None

    lsh_index_8 = faiss.IndexIDMap(faiss.IndexLSH(out_feats, out_feats*8))
    lsh_index_4 = faiss.IndexIDMap(faiss.IndexLSH(out_feats, out_feats*4))
    lsh_index_2 = faiss.IndexIDMap(faiss.IndexLSH(out_feats, out_feats*2))

    baseline = []
    filling_baseline = False
    for faiss_index, faiss_metric in (
        ("IDMap,Flat", faiss.METRIC_L2),
        (lsh_index_8, "IDMap,IndexLSH{}".format(out_feats*16)),
        (lsh_index_8, "IDMap,IndexLSH{}".format(out_feats*8)),
        (lsh_index_4, "IDMap,IndexLSH{}".format(out_feats*4)),
        (lsh_index_2, "IDMap,IndexLSH{}".format(out_feats*2)),
        # # ("IDMap,Flat", faiss.METRIC_INNER_PRODUCT),
        # ("IDMap,HNSW64,Flat", faiss.METRIC_L2),
        # ("IDMap,HNSW128,Flat", faiss.METRIC_L2),
        # # ("IDMap,HNSW64,Flat", faiss.METRIC_INNER_PRODUCT),
        # # ("IDMap,HNSW128,Flat", faiss.METRIC_INNER_PRODUCT),
        # ("IDMap,IVF20,Flat", faiss.METRIC_L2),
        # ("IDMap,IVF55,Flat", faiss.METRIC_L2),
        # # ("IDMap,IVF20,Flat", faiss.METRIC_INNER_PRODUCT),
        # # ("IDMap,IVF55,Flat", faiss.METRIC_INNER_PRODUCT),
        # ("IDMap,IVF500_HNSW32,Flat", faiss.METRIC_L2),
        # ("IDMap,PQ16", faiss.METRIC_L2),
        # ("IDMap,PQ8x8", faiss.METRIC_L2),
        # ("IDMap,PCA32,IVF20_PQ8x8,Flat", faiss.METRIC_L2),
        # ("IDMap,IVF55,SQ4", faiss.METRIC_L2),
        # ("IDMap,IVF55,SQ8", faiss.METRIC_L2),
        # ("IDMap,IVF55,PQ8+8", faiss.METRIC_L2),
        # ("IDMap,PCA32,IVF55,PQ8+8", faiss.METRIC_L2),
    ):

        if not baseline:
            filling_baseline = True

        del index

        if isinstance(faiss_index, str):
            logging.info("Using index %s with metric %s", faiss_index, faiss_metric)
            try:
                index = faiss.index_factory(out_feats, faiss_index, faiss_metric)
            except RuntimeError as e:
                logging.error("Invalid configuration: %s", str(e))
                continue
        else:
            index = faiss_index
            faiss_index = faiss_metric  # store in variable to save in out file
            logging.info("Using index %s", faiss_index)

        logging.info("Extracting features for train batch #1...")
        feat_iter = feature_iterator(
            model, dataset_train, batch_size=args.max_train or 200)
        feats, labels = next(feat_iter)
        logging.debug("First batch shape %s", feats.shape)

        logging.debug(
            "Memory usage before training: %s",
            humanize.naturalsize(faiss.get_mem_usage_kb() * 1024 - base_faiss_mem))

        t_train = 0
        if not index.is_trained:  # is_trained is False if method doesn't need training
            logging.info("Train index using train batch #1 with %s samples", len(feats))
            t0 = time.time()
            index.train(feats)
            t_train += (time.time() - t0)

        logging.info(
            "Index %s samples of batch #1", feats.shape[0])
        t0 = time.time()
        index.add_with_ids(feats, labels)
        t_index = (time.time() - t0)

        b_failed = None
        for i, (feats, labels) in enumerate(feat_iter, start=2):
            logging.info("Index %s samples of batch #%s", feats.shape[0], i)
            t0 = time.time()
            try:
                index.add_with_ids(feats, labels)
            except RuntimeError as e:
                logging.error("Error during indexing: %s", str(e))
                b_failed = e
                break

            t_index += (time.time() - t0)

        if b_failed is not None:
            raise b_failed

        logging.info("Extracting test features...")
        t_search = 0
        top_hits = collections.defaultdict(int)
        search_hits = []
        n_logos = 0
        for i, (feats, labels) in enumerate(
                feature_iterator(
                    model,
                    dataset_test,
                    batch_size=args.batch_predict or 100),
                start=1):
            logging.info(
                "Search batch #%s (contains %s feats of length %s)",
                i, len(feats), len(feats[0]))

            t0 = time.time()
            ds, ids = index.search(feats, args.k)
            t_search += (time.time() - t0)

            sorted_by_distance = list(
                list(
                    map(lambda x: x[1],
                        sorted(zip(ind_ds, ind_ids), key=lambda x: x[0])
                        )
                ) for ind_ds, ind_ids in zip(ds, ids))

            for true_label, candidates in zip(labels, sorted_by_distance):
                top_candidates = candidates[:args.k]

                if not len(top_candidates):
                    logging.warning("Image with no candidates")

                for ith, (candidate, distance) in top_candidates:
                    if true_label == candidate:
                        for k in range(ith, args.k):
                            top_hits[k] += 1

                baseline_res = sorted_by_distance if filling_baseline else baseline[i]
                search_hits.append(np.in1d(baseline_res, sorted_by_distance))

                n_logos += 1

            if filling_baseline:
                baseline.append(sorted_by_distance)

        dt_train = datetime.timedelta(seconds=t_train)
        dt_index = datetime.timedelta(seconds=t_index)
        dt_search = datetime.timedelta(seconds=t_search)
        mem_faiss_proc = faiss.get_mem_usage_kb() * 1024
        mem_faiss_idx_ram = mem_faiss_proc - base_faiss_mem
        mem_faiss_idx_disk = get_memory(index)

        logging.info(
            "Faiss Process Memory: %s", humanize.naturalsize(mem_faiss_proc))
        logging.info(
            "Index RAM Memory: %s", humanize.naturalsize(mem_faiss_idx_ram))
        logging.info(
            "Index Disk Memory: %s", humanize.naturalsize(mem_faiss_idx_disk))

        # Accuracies
        accuracies_label = collections.defaultdict(float)
        for k, val in top_hits.items():
            acc = val / n_logos
            logging.info("Top-{} Accuracy: %.4f", k, acc)
            accuracies_label[k] = acc

        logging.info("Train time %s", str(dt_train))
        logging.info("Index time %s", str(dt_index))
        logging.info("Search time %s", str(dt_search))

        if args.out is not None:
            args.out.write(
                "{}\t{}\t{:.4f}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                faiss_index, faiss_metric,
                '\t'.join(['{:.4f}'.format(acc) for acc in accuracies_label]),
                t_train, t_index, t_search,
                mem_faiss_proc, mem_faiss_idx_ram, mem_faiss_idx_disk
                )
            )
            args.out.flush()

        filling_baseline = False


    args.out.close()


if __name__ == '__main__':
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    main()
