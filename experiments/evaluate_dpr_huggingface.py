# Imports
import sys
import os
import argparse
import json
import random
import numpy as np
import time
import datetime
import pandas as pd
import torch
from transformers import BertTokenizer, DistilBertTokenizer, ElectraTokenizer
from datasets import Dataset

# Own imports
from custom_dpr import DPRQuestionEncoder, DPRContextEncoder

# Index for models
model_index = {
    'bert': ('bert-base-uncased', BertTokenizer),
    'distilbert': ('distilbert-base-uncased', DistilBertTokenizer),
    'electra': ('google/electra-small-discriminator', ElectraTokenizer)
}


def preprocess_question_dataset(dataset_instance, question_tokenizer, max_seq_length):
    """
    Function for tokenizing the question dataset, used with Huggingface map function.
    Inputs:
        dataset_instance - Instance from the Huggingface dataset
        question_tokenizer - Tokenizer instance to use for encoding the questions
        max_seq_length - Maximum sequence length for truncation
    Outputs:
        dataset_instance - HuggingFace dataset instance augmented with encodings
    """

    # Encode the question
    question_inputs = question_tokenizer(
        dataset_instance['question'],
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt',
    )
    question_ids = question_inputs['input_ids'].squeeze()
    question_mask = question_inputs['attention_mask'].squeeze()

    # Return the new columns
    return {
        'question_ids': question_ids,
        'question_mask': question_mask
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
    correct_passage_ids = []
    all_passages = []
    all_passage_ids = []

    # Get all instances from the dataset
    for instance in dataset:
        questions.append(instance['question'])
        correct_passage_ids.append(instance['positive_ctxs'][0]['passage_id'])

        if args.dont_embed_title:
            all_passages.append(instance['positive_ctxs'][0]['text'])
        else:
            all_passages.append(instance['positive_ctxs'][0]['title'] + ' ' + instance['positive_ctxs'][0]['text'])
        all_passage_ids.append(instance['positive_ctxs'][0]['passage_id'])

        for neg_context in (instance['hard_negative_ctxs'] + instance['negative_ctxs']):
            if args.dont_embed_title:
                all_passages.append(neg_context['text'])
            else:
                all_passages.append(neg_context['title'] + ' ' + neg_context['text'])
            all_passage_ids.append(neg_context['passage_id'])

    # Create a pandas DataFrame for the questions
    question_df = pd.DataFrame(list(zip(questions, correct_passage_ids)), columns=['question', 'correct_passage_id'])

    # Create a pandas DataFrame from the passages and remove duplicates
    passage_df = pd.DataFrame(list(zip(all_passages, all_passage_ids)), columns=['passage', 'passage_id'])
    passage_df.drop_duplicates(subset=['passage_id'])

    # Create Huggingface Dataset from the dataframes
    question_dataset = Dataset.from_pandas(question_df)
    passage_dataset = Dataset.from_pandas(passage_df)

    # Return the datasets
    return question_dataset, passage_dataset


def calculate_accuracy_mrr(correct_passage_id, retrieved_passage_ids):
    """
    Function that calculates the accuracy and mrr for a given query.
    Inputs:
        correct_passage_id - Id of the passage that is marked as the answer
        retrieved_passage_ids - List of retrieved passage ids
    Outputs:
        top_1_acc - Int (either 1 or 0) indicating if the answer is in the top 1
        top_20_acc - Int (either 1 or 0) indicating if the answer is in the top 20
        top_100_acc - Int (either 1 or 0) indicating if the answer is in the top 100
        mrr - MRR of the current instance
    """

    # Calculate the top 1 accuracy
    if correct_passage_id == retrieved_passage_ids[0]:
        top_1_acc = 1
    else:
        top_1_acc = 0
    
    # Calculate the top 20 accuracy
    if correct_passage_id in retrieved_passage_ids[:20]:
        top_20_acc = 1
    else:
        top_20_acc = 0
    
    # Calculate the top 100 accuracy
    if correct_passage_id in retrieved_passage_ids:
        top_100_acc = 1
    else:
        top_100_acc = 0

    # Calculate the MRR
    try:
        position = retrieved_passage_ids.index(correct_passage_id)
        mrr = 1 / (position + 1)
    except ValueError:
        mrr = 0

    # Return the accuracies and mrr
    return top_1_acc, top_20_acc, top_100_acc, mrr


def evaluate_model(args, device):
    """
    Function for evaluating the model.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to train on
    """

    train_filename = 'nq-train.json'
    dev_filename = 'nq-dev.json'

    # Load the dataset
    print('Loading data..')
    question_dataset, passage_dataset = load_dataset(args, args.data_dir + dev_filename)
    print('Data loaded')

    # Load the model
    print('Loading model..')
    trained_location = args.load_dir + args.model + str(args.max_seq_length) + "/"
    model_location, tokenizer_class = model_index[args.model]
    question_tokenizer = tokenizer_class.from_pretrained(model_location)
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_encoder.question_encoder.replace_bert(args.model, trained_location + 'question_encoder/', 0.0)
    context_tokenizer = tokenizer_class.from_pretrained(model_location)
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_encoder.ctx_encoder.replace_bert(args.model, trained_location + 'context_encoder/', 0.0)
    print('Model loaded')

    # Set models to evaluation
    question_encoder.eval()
    context_encoder.eval()

    # Encode the questions
    print('Encoding questions..')
    encode_question_time_start = time.time()
    question_dataset = question_dataset.map(lambda example: {'embeddings': question_encoder(**question_tokenizer(example["question"], padding='max_length', truncation=True, max_length=args.max_seq_length, return_tensors="pt"))[0][0].detach().numpy()})
    question_dataset.set_format(type='numpy', columns=['embeddings'], output_all_columns=True)
    encode_question_time_stop = time.time()
    encode_question_time_elapsed = encode_question_time_stop - encode_question_time_start
    average_encode_question_time = encode_question_time_elapsed / len(question_dataset)
    print('Questions encoded. Elapsed time: {}'.format(str(datetime.timedelta(seconds=encode_question_time_elapsed))))

    # Encode the passages
    print('Encoding passages..')
    encode_passage_time_start = time.time()
    passage_dataset = passage_dataset.map(lambda example: {'embeddings': context_encoder(**context_tokenizer(example["passage"], padding='max_length', truncation=True, max_length=args.max_seq_length, return_tensors="pt"))[0][0].detach().numpy()})
    passage_dataset.set_format(type='numpy', columns=['embeddings'], output_all_columns=True)
    passage_dataset.add_faiss_index(column='embeddings')
    encode_passage_time_stop = time.time()
    encode_passage_time_elapsed = encode_passage_time_stop - encode_passage_time_start
    average_encode_passage_time = encode_passage_time_elapsed / len(passage_dataset)
    print('Passages encoded. Elapsed time: {}'.format(str(datetime.timedelta(seconds=encode_passage_time_elapsed))))

    # Evaluate the model
    num_questions = len(question_dataset)
    inference_times = []
    top_1_acc = 0
    top_20_acc = 0
    top_100_acc = 0
    total_mrr = 0
    print('Starting evaluation..')
    for question in question_dataset:
        # Retrieve the question embedding
        question_embedding = question['embeddings']
        
        # Get the nearest passages
        inference_time_start = time.time()
        _, retrieved_passages = passage_dataset.get_nearest_examples('embeddings', question_embedding, k=100)
        inference_time_stop = time.time()
        inference_time_elapsed = inference_time_stop - inference_time_start
        inference_times.append(inference_time_elapsed)

        # Calculate the accuracy
        top_1_correct, top_20_correct, top_100_correct, mrr = calculate_accuracy_mrr(question['correct_passage_id'], retrieved_passages['passage_id'])
        top_1_acc += top_1_correct
        top_20_acc += top_20_correct
        top_100_acc += top_100_correct
        total_mrr += mrr
    top_1_acc = top_1_acc / num_questions
    top_20_acc = top_20_acc / num_questions
    top_100_acc = top_100_acc / num_questions
    total_mrr = total_mrr / num_questions
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    print('Evaluation finished.')
    print('Inference time.    Average: {}    Std: {}'.format(mean_inference_time, std_inference_time))
    print('Accuracy@1: {}    Accuracy@20: {}    Accuracy@100: {}    MRR: {}'.format(top_1_acc, top_20_acc, top_100_acc, total_mrr))

    # Save the results
    print('Saving results..')
    df = pd.DataFrame(
        [[average_encode_question_time, average_encode_passage_time, mean_inference_time, std_inference_time, top_1_acc, top_20_acc, top_100_acc, total_mrr]], 
        columns=['Mean encode question time', 'Mean encode passage time', 'Mean inference time', 'Std inference time', 'Acc@1', 'Acc@20', 'Acc@100', 'MRR']
    )
    df.to_csv(args.output_dir + args.model + str(args.max_seq_length) + '_metrics.csv')
    print('Results saved')


def main(args):
    """
    Function for handling the arguments and starting the evaluation.
    Inputs:
        args - Namespace object from the argument parser
    """

    # Set a seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----EVALUATION PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    print('Loading directory: {}'.format(args.load_dir + args.model))
    print('Maximum sequence length: {}'.format(args.max_seq_length))
    print('Model evaluation output directory: {}'.format(args.output_dir))
    print('Embed title: {}'.format(not args.dont_embed_title))
    print('Seed: {}'.format(args.seed))
    print('-----------------------------')

    # Start evaluation
    evaluate_model(args, device)


# Command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='bert', type=str,
                        help='What model to use. Default is bert.',
                        choices=['bert', 'distilbert', 'electra'])
    parser.add_argument('--load_dir', default='saved_models/', type=str,
                        help='Directory for loading the trained models. Default is saved_models/.')
    parser.add_argument('--max_seq_length', default=256, type=int,
                        help='Maximum sequence length. Default is 256.')
    
    # DPR hyperparameters
    parser.add_argument('--dont_embed_title', action='store_true',
                        help='Do not embed titles. Titles are embedded by default.')

    # Evaluation hyperparameters
    parser.add_argument('--data_dir', default='data/downloads/data/retriever/', type=str,
                        help='Directory where the data is stored. Default is data/downloads/data/retriever/.')
    parser.add_argument('--output_dir', default='evaluation_outputs/', type=str,
                        help='Directory for saving the model evaluation metrics. Default is evaluation_outputs/.')
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use during training. Default is 1234.')

    # Parse the arguments
    args = parser.parse_args()

    # Train the model
    main(args)