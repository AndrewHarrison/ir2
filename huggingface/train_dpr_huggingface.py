# Imports
import sys
import os
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, DistilBertTokenizer, ElectraTokenizer
from custom_dpr import DPRQuestionEncoder, DPRContextEncoder
from combined_dpr_model import DPR

# DEBUG
sys.stdout.reconfigure(encoding='utf-8')

# Index for models
model_index = {
    'bert': ('bert-base-uncased', BertTokenizer),
    'distilbert': ('distilbert-base-uncased', DistilBertTokenizer),
    'electra': ('google/electra-small-discriminator', ElectraTokenizer)
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
    gold_contexts = []
    neg_contexts = []
    # Get all instances from the dataset
    for instance in dataset:
        questions.append(instance['question'])
        if args.dont_embed_title:
            gold_contexts.append(instance['positive_ctxs'][0]['text'])
        else:
            gold_contexts.append(instance['positive_ctxs'][0]['title'] + ' ' + instance['positive_ctxs'][0]['text'])
        if instance['hard_negative_ctxs']:
            if args.dont_embed_title:
                neg_contexts.append(instance['hard_negative_ctxs'][0]['text'])
            else:
                neg_contexts.append(instance['hard_negative_ctxs'][0]['title'] + ' ' + instance['hard_negative_ctxs'][0]['text'])
        else:
            if args.dont_embed_title:
                neg_contexts.append(instance['negative_ctxs'][0]['text'])
            else:
                neg_contexts.append(instance['negative_ctxs'][0]['title'] + ' ' + instance['negative_ctxs'][0]['text'])

    # Return the questions, gold contexts and negative contexts
    return questions, gold_contexts, neg_contexts


def shuffle_data(train_query_encodings, train_gold_context_encodings, train_neg_context_encodings):
    """
    Function that shuffles the training data.
    Inputs:
        train_query_encodings - Encodings of the training queries
        train_gold_context_encodings - Encodings of the training gold contexts
        train_neg_context_encodings - Encodings of the training negative contexts
    Outputs:
        shuffled_query_encodings - Shuffled encodings of the training queries
        shuffled_gold_encodings - Shuffled encodings of the training gold contexts
        shuffled_neg_encodings - Shuffled encodings of the training negative contexts
    """

    shuffled_query_encodings = {}
    shuffled_gold_encodings = {}
    shuffled_neg_encodings = {}

    # Get all the attributes
    query_input_ids = train_query_encodings.input_ids
    query_attention_mask = train_query_encodings.attention_mask

    gold_context_input_ids = train_gold_context_encodings.input_ids
    gold_context_attention_mask = train_gold_context_encodings.attention_mask

    neg_context_input_ids = train_neg_context_encodings.input_ids
    neg_context_attention_mask = train_neg_context_encodings.attention_mask

    # Generate a random ordering
    idx = torch.randperm(query_input_ids.size()[0])

    # Order the data with the new ordering
    shuffled_query_encodings['input_ids'] = query_input_ids[idx]
    shuffled_query_encodings['attention_mask'] = query_attention_mask[idx]

    shuffled_gold_encodings['input_ids'] = gold_context_input_ids[idx]
    shuffled_gold_encodings['attention_mask'] = gold_context_attention_mask[idx]

    shuffled_neg_encodings['input_ids'] = neg_context_input_ids[idx]
    shuffled_neg_encodings['attention_mask'] = neg_context_attention_mask[idx]

    # Return the shuffled data
    return shuffled_query_encodings, shuffled_gold_encodings, shuffled_neg_encodings


def get_batch(device, num_questions, train_query_encodings, train_gold_context_encodings, train_neg_context_encodings):
    """
    Function for generating a batch.
    Inputs:
        device - PyTorch device to train on
        num_questions - Number of questions in the training set
        train_query_encodings - Encodings of the training queries
        train_gold_context_encodings - Encodings of the training gold contexts
        train_neg_context_encodings - Encodings of the training negative contexts
    Outputs:
        context_input_ids_tensor - Tensor containing context input ids
        context_attention_mask_tensor - Tensor containing context attention masks
        query_input_ids - Tensor containing query input ids
        query_attention_mask - Tensor containing query attention masks
        true - Whether the context is positive or negative
    """

    true = [0, 2]
    context_input_ids_tensor = []
    context_attention_mask_tensor = []

    # Selecting a random query with its positive context and negative context
    idx = random.randint(0, num_questions-1)
    gold_context_input_ids = train_gold_context_encodings['input_ids'][idx]
    gold_context_attention_mask = train_gold_context_encodings['attention_mask'][idx]
    context_input_ids_tensor.append(gold_context_input_ids)
    context_attention_mask_tensor.append(gold_context_attention_mask)

    query_input_ids = train_query_encodings['input_ids'][idx]
    query_attention_mask = train_query_encodings['attention_mask'][idx]

    neg_context_input_ids = train_neg_context_encodings['input_ids'][idx]
    neg_context_attention_mask = train_neg_context_encodings['attention_mask'][idx]
    context_input_ids_tensor.append(neg_context_input_ids)
    context_attention_mask_tensor.append(neg_context_attention_mask)

    # DEBUG
    idx = random.randint(0, num_questions-1)
    gold_context_input_ids = train_gold_context_encodings['input_ids'][idx]
    gold_context_attention_mask = train_gold_context_encodings['attention_mask'][idx]
    context_input_ids_tensor.append(gold_context_input_ids)
    context_attention_mask_tensor.append(gold_context_attention_mask)

    query_input_ids = train_query_encodings['input_ids'][idx]
    query_attention_mask = train_query_encodings['attention_mask'][idx]

    neg_context_input_ids = train_neg_context_encodings['input_ids'][idx]
    neg_context_attention_mask = train_neg_context_encodings['attention_mask'][idx]
    context_input_ids_tensor.append(neg_context_input_ids)
    context_attention_mask_tensor.append(neg_context_attention_mask)

    # Stack into tensors
    context_input_ids_tensor = torch.stack(context_input_ids_tensor)
    context_attention_mask_tensor = torch.stack(context_attention_mask_tensor)
    query_input_ids = query_input_ids.unsqueeze(0)
    query_attention_mask = query_attention_mask.unsqueeze(0)

    # Return the context inputs and query inputs
    return context_input_ids_tensor, context_attention_mask_tensor, query_input_ids, query_attention_mask, torch.tensor(true, device=device).unsqueeze(0)


def perform_training_epoch(dpr_model, device, batch_size, optimizer, criterion, train_query_encodings, train_gold_context_encodings, train_neg_context_encodings):
    """
    Function that performs a single training epoch.
    Inputs:
        dpr_model - DPR model instance that is trained
        device - PyTorch device to train on
        batch_size - Size of the training batches
        optimizer - PyTorch optimizer instance
        criterion - PyTorch loss function instance
        train_query_encodings - Encodings of the training queries
        train_gold_context_encodings - Encodings of the training gold contexts
        train_neg_context_encodings - Encodings of the training negative contexts
    Outputs:
        epoch_loss - Average loss over the epoch
    """

    epoch_loss = 0

    # Shuffle the data
    train_query_encodings, train_gold_context_encodings, train_neg_context_encodings = shuffle_data(train_query_encodings, train_gold_context_encodings, train_neg_context_encodings)

    # Calculate the number of batches
    num_questions = len(train_gold_context_encodings['input_ids'])
    num_batches = int(num_questions / batch_size) + (num_questions % batch_size > 0)

    # Loop over the batches
    for i in range(1, num_batches + 1):
        # Remove gradients from the optimizer
        optimizer.zero_grad()

        # Forward the batch through the models
        context_input_ids_tensor, context_attention_mask_tensor, query_input_ids, query_attention_mask, true = get_batch(device, num_questions, train_query_encodings, train_gold_context_encodings, train_neg_context_encodings)
        pred = dpr_model(context_input_ids_tensor, context_attention_mask_tensor,query_input_ids,query_attention_mask)

        # Calcualte the loss
        print(pred.size())
        print(true.size())
        loss = criterion(pred, true)

        # Backwards pass
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    
    # Return the loss
    return epoch_loss / num_questions


def train_model(args, device):
    """
    Function for training the model.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to train on
    Outputs:
        ?
    """

    train_filename = 'nq-train.json'
    dev_filename = 'nq-dev.json'

    # Load the dataset
    print('Loading data..')
    train_questions, train_gold_contexts, train_neg_contexts = load_dataset(args, args.data_dir + train_filename)
    print('Data loaded')

    # Load the model
    print('Loading model..')
    model_location, tokenizer_class = model_index[args.model]
    question_tokenizer = tokenizer_class.from_pretrained(model_location)
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_encoder.question_encoder.replace_bert(args.model, model_location)
    context_tokenizer = tokenizer_class.from_pretrained(model_location)
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_encoder.ctx_encoder.replace_bert(args.model, model_location)
    print('Model loaded')

    # Combine into a single DPR model
    dpr_model = DPR(question_encoder = question_encoder,
                context_encoder = context_encoder, 
                question_tokenizer = question_tokenizer, 
                context_tokenizer = context_tokenizer)
    dpr_model.to(device)
    dpr_model.train()
    criterion = nn.NLLLoss().to(device)
    optimizer = torch.optim.AdamW(dpr_model.parameters(), lr = 10e-5)

    # Encode the training data
    print('Encoding training data..')
    train_query_encodings = question_tokenizer(train_questions, truncation=True, padding='max_length', return_tensors = 'pt').to(device)
    train_gold_context_encodings = context_tokenizer(train_gold_contexts, truncation=True, padding='max_length', return_tensors = 'pt').to(device)
    train_neg_context_encodings = context_tokenizer(train_neg_contexts, truncation=True, padding='max_length', return_tensors = 'pt').to(device)
    print('Training data encoded')

    # Train the model
    print('Starting training..')
    for epoch in range(1, args.n_epochs + 1):
        average_epoch_loss = perform_training_epoch(
            dpr_model = dpr_model,
            device = device,
            batch_size = args.batch_size, 
            optimizer = optimizer, 
            criterion = criterion,
            train_query_encodings = train_query_encodings,
            train_gold_context_encodings = train_gold_context_encodings,
            train_neg_context_encodings = train_neg_context_encodings)
        # Report on the loss
        print('Epoch : {}  Loss : {}'.format(epoch, average_epoch_loss))
    print('Training finished')

    # Save the model
    print('Saving model..')
    dpr_model.question_encoder.question_encoder.bert_model.save_pretrained(args.save_dir + args.model + '/question_encoder/')
    dpr_model.context_encoder.ctx_encoder.bert_model.save_pretrained(args.save_dir + args.model + '/context_encoder/')
    print('Model saved')


def main(args):
    """
    Function for handling the arguments and starting the training.
    Inputs:
        args - Namespace object from the argument parser
    """

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----TRAINING PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    print('Max query length: {}'.format(args.max_seq_len_query))
    print('Max context passage length: {}'.format(args.max_seq_len_passage))
    print('Num training epochs: {}'.format(args.n_epochs))
    print('Batch size: {}'.format(args.batch_size))
    print('Gradient accumulation steps: {}'.format(args.grad_acc_steps))
    print('Model saving directory: {}'.format(args.save_dir))
    print('Step interval for evaluation: {}'.format(args.evaluate_every))
    print('Num positive contexts: {}'.format(args.num_positives))
    print('Num negative contexts: {}'.format(args.num_hard_negatives))
    print('Embed title: {}'.format(not args.dont_embed_title))
    print('-----------------------------')

    # Start training
    train_model(args, device)


# Command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='bert', type=str,
                        help='What model to use. Default is bert.',
                        choices=['bert', 'distilbert', 'electra'])
    parser.add_argument('--max_seq_len_query', default=64, type=int,
                        help='Maximum length of query sequence. Default is 64.')
    parser.add_argument('--max_seq_len_passage', default=256, type=int,
                        help='Maximum length of context passage. Default is 256.')

    # Training hyperparameters
    parser.add_argument('--data_dir', default='data/downloads/data/retriever/', type=str,
                        help='Directory where the data is stored. Default is data/downloads/data/retriever/.')
    parser.add_argument('--n_epochs', default=2, type=int,
                        help='Number of epochs to train for. Default is 2.')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Training batch size. Default is 2.')
    parser.add_argument('--grad_acc_steps', default=8, type=int,
                        help='Number of gradient accumulation steps. Default is 8.')
    parser.add_argument('--save_dir', default='saved_models/', type=str,
                        help='Directory for saving the models. Default is saved_models/.')
    parser.add_argument('--evaluate_every', default=3000, type=int,
                        help='Step interval for evaluation. Default is 3000.')
    parser.add_argument('--num_positives', default=1, type=int,
                        help='Number of positive contexts per question. Default is 1.')
    parser.add_argument('--num_hard_negatives', default=1, type=int,
                        help='Number of negative contexts per question. Default is 1.')
    parser.add_argument('--dont_embed_title', action='store_true',
                        help='Do not embed titles. Titles are embedded by default.')

    # Parse the arguments
    args = parser.parse_args()

    # Train the model
    main(args)