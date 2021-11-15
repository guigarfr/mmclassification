import argparse
import logging
import os

import faiss

from rpmlbaselib import storage as st

import rplogo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a library faiss index from a faiss pickle.')
    parser.add_argument(
        'faiss_model',
        default=None,
        type=argparse.FileType("rb"),
        help='Faiss pickled model.')
    parser.add_argument(
        '--version',
        help='Specify the version to set to faiss',
        default='0.0',
    )
    parser.add_argument(
        '--use-gpu',
        help='Specify to use gpu or not with model',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--index-type',
        help='Specify the index type',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--out',
        default='converted.rpml',
        type=argparse.FileType("wb+"),
        help='Faiss pickled model.')

    args = parser.parse_args()

    # Check additional argument conditions
    splitted_version = args.version.split('.')
    if 1 < len(splitted_version) > 2:
        raise ValueError("Invalid version")

    # won't need these opened
    args.out.close()
    args.faiss_model.close()

    return args


def main():
    args = parse_args()

    model = faiss.read_index(args.faiss_model.name)

    if not isinstance(model, faiss.Index):
        raise ValueError("Invalid pickled index.")

    lib_model = rplogo.methods.faiss_classifier.FaissClassifier()
    lib_model.model = model
    lib_model.use_gpu = args.use_gpu
    lib_model.input_features = model.d
    lib_model.metric = model.metric_type
    lib_model.set_version(*map(int, args.version.split('.')))

    filename = os.path.basename(args.out.name)
    path = os.path.dirname(args.out.name)
    logging.debug("File %s in %s", filename, path)
    lib_model.save(
        filename,
        storage=st.LocalStorage(root=path or None)
    )


if __name__ == '__main__':
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    main()
