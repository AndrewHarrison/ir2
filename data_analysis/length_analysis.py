# Imports
import sys
import os
import argparse
import json
import random
import time
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, DistilBertTokenizer, ElectraTokenizer, DPRConfig
from datasets import Dataset

# Index for models
model_index = {
    'bert': ('bert-base-uncased', BertTokenizer),
    'distilbert': ('distilbert-base-uncased', DistilBertTokenizer),
    'electra': ('google/electra-small-discriminator', ElectraTokenizer),
    'tinybert': ('huawei-noah/TinyBERT_General_4L_312D', BertTokenizer)
}


def preprocess_dataset(dataset_instance, tokenizer, content):
    """
    Function for tokenizing the dataset, used with Huggingface map function.
    Inputs:
        dataset_instance - Instance from the Huggingface dataset
        tokenizer - Tokenizer instance to use for tokenization
        content - Attribute that contains the content of the instance
    Outputs:
        dict - Dict containing the new added columns
    """

    # Tokenize the instance
    tokenized = question_tokenizer(
        dataset_instance[content],
        return_tensors='pt',
    )
    token_ids = tokenized['input_ids'].squeeze()

    # Get the length of the tokens
    token_length = token_ids.size[0]

    # Return the new columns
    return {
        'length': token_length
    }


def load_dataset(args, path):
    """
    Function for loading the given dataset.
    Inputs:
        args - Namespace object from the argument parser
        path - String representing the location of the .json file
    Outputs:
        questions - List of questions from the dataset
        gold_contexts - List of contexts that are the answer to the question
        neg_contexts - List of contexts that are NOT the answer to the question
    """

    # Read the file
    with open(path, 'rb') as f:
        dataset = json.load(f)

    questions = []
    all_passages = []

    # Get all instances from the dataset
    for instance in dataset:
        questions.append(instance['question'])

        if args.dont_embed_title:
            all_passages.append(instance['positive_ctxs'][0]['text'])
        else:
            all_passages.append(instance['positive_ctxs'][0]['title'] + ' ' + instance['positive_ctxs'][0]['text'])

        for neg_context in (instance['hard_negative_ctxs'] + instance['negative_ctxs']):
            if args.dont_embed_title:
                all_passages.append(neg_context['text'])
            else:
                all_passages.append(neg_context['title'] + ' ' + neg_context['text'])

    # Create a pandas DataFrame for the questions
    question_df = pd.DataFrame(questions, columns=['question'])

    # Create a pandas DataFrame from the passages and remove duplicates
    passage_df = pd.DataFrame(all_passages, columns=['passage'])
    passage_df.drop_duplicates(subset=['passage'])

    # Create Huggingface Dataset from the dataframes
    question_dataset = Dataset.from_pandas(question_df)
    passage_dataset = Dataset.from_pandas(passage_df)

    # Return the datasets
    return question_dataset, passage_dataset


def perform_analysis(args, device):
    """
    Function for performning data analysis.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to train on
    """

    train_filename = 'nq-train.json'
    dev_filename = 'nq-dev.json'

    # Load the dataset
    print('Loading data..')
    train_questions, train_passages = load_dataset(args, args.data_dir + train_filename)
    dev_question, dev_passages = load_dataset(args, args.data_dir + dev_filename)
    print('Data loaded')

    # Load the tokenizer
    print('Loading tokenizer..')
    model_location, tokenizer_class = model_index[args.model]
    question_tokenizer = tokenizer_class.from_pretrained(model_location)
    context_tokenizer = tokenizer_class.from_pretrained(model_location)
    print('Tokenizer loaded')

    # Encode the training data
    print('Calculating length of training data..')
    train_questions = train_questions.map(
        lambda x: preprocess_dataset(
            x,
            tokenizer = question_tokenizer,
            content = 'question',       
        ),
        batched=False,
    )
    train_questions_dataframe = train_questions.to_pandas()
    train_passages = train_passages.map(
        lambda x: preprocess_dataset(
            x,
            tokenizer = context_tokenizer,
            content = 'passage',       
        ),
        batched=False,
    )
    train_passages_dataframe = train_passages.to_pandas()
    print('Training data lengths calculated')

    # Encode the dev data
    print('Calculating length of dev data..')
    dev_questions = dev_questions.map(
        lambda x: preprocess_dataset(
            x,
            tokenizer = question_tokenizer,
            content = 'question',       
        ),
        batched=False,
    )
    dev_questions_dataframe = dev_questions.to_pandas()
    dev_passages = dev_passages.map(
        lambda x: preprocess_dataset(
            x,
            tokenizer = context_tokenizer,
            content = 'passage',       
        ),
        batched=False,
    )
    dev_passages_dataframe = dev_passages.to_pandas()
    print('Dev data lengths calculated')

    # TODO: DO ANALYSIS HERE 


def main(args):
    """
    Function for handling the arguments and starting the training.
    Inputs:
        args - Namespace object from the argument parser
    """

    # Set a seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----TRAINING PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    print('Embed title: {}'.format(not args.dont_embed_title))
    print('Seed: {}'.format(args.seed))
    print('-----------------------------')

    # Start analysis
    perform_analysis(args, device)


# Command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='bert', type=str,
                        help='What model to use. Default is bert.',
                        choices=['bert', 'distilbert', 'electra', 'tinybert'])
    
    # DPR hyperparameters
    parser.add_argument('--dont_embed_title', action='store_true',
                        help='Do not embed titles. Titles are embedded by default.')

    # Training hyperparameters
    parser.add_argument('--data_dir', default='data/downloads/data/retriever/', type=str,
                        help='Directory where the data is stored. Default is data/downloads/data/retriever/.')
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use during training. Default is 1234.')

    # Parse the arguments
    args = parser.parse_args()

    # Do data analysis
    main(args)
