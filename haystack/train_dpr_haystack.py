# imports
import os
import argparse
import torch
from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store import FAISSDocumentStore


def train_model(args):
    """
    Function for training the model.
    Inputs:
        args - Namespace object from the argument parser
    Outputs:
        ?
    """

    # data files
    if args.dataset == 'NQ':
        train_filename = "nq-train.json"
        dev_filename = "nq-dev.json"
    else:
        train_filename = "train/biencoder-nq-train.json"
        dev_filename = "dev/biencoder-nq-dev.json"

    # load the model
    print('Loading model..')
    document_store = FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        return_embedding=True,
        use_gpu=True)
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model=args.model,
        passage_embedding_model=args.model,
        max_seq_len_query=args.max_seq_len_query,
        max_seq_len_passage=args.max_seq_len_passage,
        use_gpu=True,
        #devices=["cuda:0"],
    )
    print('Model loaded')

    # train the model
    print('Starting training..')
    retriever.train(
        data_dir=args.data_dir,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=dev_filename,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        grad_acc_steps=args.grad_acc_steps,
        save_dir=args.save_dir + '/' + args.dataset + '/' + args.model,
        evaluate_every=args.evaluate_every,
        embed_title=not args.dont_embed_title,
        num_positives=args.num_positives,
        num_hard_negatives=args.num_hard_negatives,
        #max_sample=40, # DEBUG
        #n_gpu=2, # DEBUG
        max_processes=1, # DEBUG
    )
    print('Training finished')


def main(args):
    """
    Function for handling the arguments and starting the training.
    Inputs:
        args - Namespace object from the argument parser
    """

    # check if GPU is available
    check_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print the given parameters
    print('-----TRAINING PARAMETERS-----')
    print('Device: {}'.format(check_device))
    print('Model: {}'.format(args.model))
    print('Max query length: {}'.format(args.max_seq_len_query))
    print('Max context passage length: {}'.format(args.max_seq_len_passage))
    print('Dataset: {}'.format(args.dataset))
    print('Num training epochs: {}'.format(args.n_epochs))
    print('Batch size: {}'.format(args.batch_size))
    print('Gradient accumulation steps: {}'.format(args.grad_acc_steps))
    print('Model saving directory: {}'.format(args.save_dir))
    print('Step interval for evaluation: {}'.format(args.evaluate_every))
    print('Num positive contexts: {}'.format(args.num_positives))
    print('Num negative contexts: {}'.format(args.num_hard_negatives))
    print('Embed title: {}'.format(not args.dont_embed_title))
    print('-----------------------------')

    # start training
    train_model(args)


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyperparameters
    parser.add_argument('--model', default='bert-base-uncased', type=str,
                        help='What model to use. Default is bert-base-uncased.',
                        choices=['bert-base-uncased', 'distilbert-base-uncased'])
    parser.add_argument('--max_seq_len_query', default=64, type=int,
                        help='Maximum length of query sequence. Default is 64.')
    parser.add_argument('--max_seq_len_passage', default=256, type=int,
                        help='Maximum length of context passage. Default is 256.')
    parser.add_argument('--dataset', default='DPR', type=str,
                        help='What dataset to use. Default is DPR.',
                        choices=['DPR', 'NQ'])

    # training hyperparameters
    parser.add_argument('--data_dir', default='data/downloads/data/retriever/', type=str,
                        help='Directory where the data is stored. Default is data/downloads/data/retriever/.')
    parser.add_argument('--n_epochs', default=1, type=int,
                        help='Number of epochs to train for. Default is 1.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Training batch size. Default is 4.')
    parser.add_argument('--grad_acc_steps', default=8, type=int,
                        help='Number of gradient accumulation steps. Default is 8.')
    parser.add_argument('--save_dir', default='saved_models', type=str,
                        help='Directory for saving the models. Default is saved_models.')
    parser.add_argument('--evaluate_every', default=3000, type=int,
                        help='Step interval for evaluation. Default is 3000.')
    parser.add_argument('--num_positives', default=1, type=int,
                        help='Number of positive contexts per question. Default is 1.')
    parser.add_argument('--num_hard_negatives', default=1, type=int,
                        help='Number of negative contexts per question. Default is 1.')
    parser.add_argument('--dont_embed_title', action='store_true',
                        help='Do not embed titles. Titles are embedded by default.')

    # parse the arguments
    args = parser.parse_args()

    # train the model
    main(args)