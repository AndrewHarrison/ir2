# imports
import os
import argparse
import torch
from haystack.retriever.dense import DensePassageRetriever
from haystack.preprocessor.utils import fetch_archive_from_http
from haystack.document_store import FAISSDocumentStore


def main(args):
    """
    Function for handling the arguments and starting the download.
    Inputs:
        args - Namespace object from the argument parser
    """

    # urls for downloading the data
    s3_url_train = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"
    s3_url_dev = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"

    # download the data
    print('Downloading data..')
    fetch_archive_from_http(s3_url_train, output_dir=args.data_dir + "train/")
    fetch_archive_from_http(s3_url_dev, output_dir=args.data_dir + "dev/")
    print('Data downloaded')


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameters
    parser.add_argument('--data_dir', default='data/original_dpr/', type=str,
                        help='Directory where to store the data. Default is data/original_dpr/.')

    # parse the arguments
    args = parser.parse_args()

    # train the model
    main(args)