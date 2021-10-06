# Imports
import sys
import os
import argparse
import json
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DistilBertTokenizer, ElectraTokenizer
from datasets import Dataset

# Own imports
from custom_dpr import DPRQuestionEncoder, DPRContextEncoder
from combined_dpr_model import DPR

# Index for models
model_index = {
    'bert': ('bert-base-uncased', BertTokenizer),
    'distilbert': ('distilbert-base-uncased', DistilBertTokenizer),
    'electra': ('google/electra-small-discriminator', ElectraTokenizer)
}


def preprocess_dataset(dataset_instance, question_tokenizer, context_tokenizer):
    """
    Function for tokenizing the dataset, used with Huggingface map function.
    Inputs:
        dataset_instance - Instance from the Huggingface dataset
        question_tokenizer - Tokenizer instance to use for encoding the questions
        context_tokenizer - Tokenizer instance to use for encoding the contexts
        dataset - HuggingFace dataset instance containing the data
    Outputs:
        dataset_instance - HuggingFace dataset instance augmented with encodings
    """

    # Encode the question
    question_inputs = question_tokenizer(
        dataset_instance['question'],
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    question_ids = question_inputs['input_ids'].squeeze()
    question_mask = question_inputs['attention_mask'].squeeze()

    # Encode the gold context
    gold_inputs = context_tokenizer(
        dataset_instance['gold_context'],
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    gold_ids = gold_inputs['input_ids'].squeeze()
    gold_mask = gold_inputs['attention_mask'].squeeze()

    # Encode the negative context
    neg_inputs = context_tokenizer(
        dataset_instance['neg_context'],
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    neg_ids = neg_inputs['input_ids'].squeeze()
    neg_mask = neg_inputs['attention_mask'].squeeze()

    # Return the new columns
    return {
        'question_ids': question_ids,
        'question_mask': question_mask,
        'gold_context_ids': gold_ids,
        'gold_context_mask': gold_mask,
        'neg_context_ids': neg_ids,
        'neg_context_mask': neg_mask
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
    
    # Create a pandas DataFrame from the dataset
    df = pd.DataFrame(list(zip(questions, gold_contexts, neg_contexts)), columns=['question', 'gold_context', 'neg_context'])

    # Create Huggingface Dataset from the dataframe
    dataset = Dataset.from_pandas(df)

    # Return the dataset
    return dataset


def perform_training_epoch(dpr_model, device, dataloader, optimizer, scheduler, criterion, num_questions):
    """
    Function that performs a single training epoch.
    Inputs:
        dpr_model - DPR model instance that is trained
        device - PyTorch device to train on
        dataloader - Dataloader instance containing the data
        optimizer - PyTorch optimizer instance
        scheduler - PyTorch scheduler instance
        criterion - PyTorch loss function instance
        num_questions - Total number of questions (dataset length)
    Outputs:
        epoch_loss - Average loss over the epoch
    """

    epoch_loss = 0

    # Loop over the batches
    for batch_index, batch in enumerate(dataloader):
        # Remove gradients from the optimizer
        optimizer.zero_grad()

        # Get the required variables from the batch
        question_ids = batch['question_ids'].to(device)
        question_mask = batch['question_mask'].to(device)
        gold_context_ids = batch['gold_context_ids'].to(device)
        gold_context_mask = batch['gold_context_mask'].to(device)
        neg_context_ids = batch['neg_context_ids'].to(device)
        neg_context_mask = batch['neg_context_mask'].to(device)

        # Forward the batch through the models
        pred = dpr_model(
            torch.cat([gold_context_ids, neg_context_ids], dim=0).to(device),
            torch.cat([gold_context_mask, neg_context_mask], dim=0).to(device),
            question_ids,
            question_mask,
        )

        # Create the truth labels
        true = list(range(0, batch['question_ids'].size()[0]))
        true = torch.tensor(true, device=device)

        # Calcualte the loss
        loss = criterion(pred, true)

        # Backwards pass
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        scheduler.step()
    
    # Return the average epoch loss
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
    train_dataset = load_dataset(args, args.data_dir + train_filename)
    print('Data loaded')

    # Load the model
    print('Loading model..')
    model_location, tokenizer_class = model_index[args.model]
    question_tokenizer = tokenizer_class.from_pretrained(model_location)
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_encoder.question_encoder.replace_bert(args.model, model_location, args.dropout)
    context_tokenizer = tokenizer_class.from_pretrained(model_location)
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_encoder.ctx_encoder.replace_bert(args.model, model_location, args.dropout)
    print('Model loaded')

    # Combine into a single DPR model
    dpr_model = DPR(question_encoder = question_encoder,
                context_encoder = context_encoder, 
                question_tokenizer = question_tokenizer, 
                context_tokenizer = context_tokenizer)
    dpr_model.to(device)
    dpr_model.train()
    criterion = nn.NLLLoss().to(device)
    optimizer = torch.optim.AdamW(dpr_model.parameters(), lr=args.lr, eps=1e-8)

    # Create the linear learning rate scheduler
    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )
    scheduler = LambdaLR(optimizer, lr_lambda, args.n_epochs)

    # Encode the training data
    print('Encoding training data..')
    train_dataset = train_dataset.map(
        lambda x: preprocess_dataset(
            x,
            question_tokenizer = question_tokenizer,
            context_tokenizer = context_tokenizer,
        ),
        batched=False,
    )
    train_dataset.set_format(type='pt', columns=['question_ids', 'question_mask', 'gold_context_ids', 'gold_context_mask', 'neg_context_ids', 'neg_context_mask'])
    len_dataset = len(train_dataset)
    print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print('Training data encoded')

    # Train the model
    print('Starting training..')
    for epoch in range(1, args.n_epochs + 1):
        average_epoch_loss = perform_training_epoch(
            dpr_model = dpr_model,
            device = device,
            dataloader = train_loader, 
            optimizer = optimizer,
            scheduler = scheduler, 
            criterion = criterion,
            num_questions = len_dataset,
        )
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

    # Set a random seed
    torch.seed()

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----TRAINING PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    print('Learning rate: {}'.format(args.lr))
    print('Dropout rate: {}'.format(args.dropout))
    print('Num training epochs: {}'.format(args.n_epochs))
    print('Batch size: {}'.format(args.batch_size))
    print('Model saving directory: {}'.format(args.save_dir))
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
    
    # DPR hyperparameters
    parser.add_argument('--dont_embed_title', action='store_true',
                        help='Do not embed titles. Titles are embedded by default.')

    # Training hyperparameters
    parser.add_argument('--data_dir', default='data/downloads/data/retriever/', type=str,
                        help='Directory where the data is stored. Default is data/downloads/data/retriever/.')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate to use during training. Default is 1e-5.')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate to use during training. Default is 0.1.')
    parser.add_argument('--n_epochs', default=2, type=int,
                        help='Number of epochs to train for. Default is 2.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Training batch size. Default is 4.')
    parser.add_argument('--save_dir', default='saved_models/', type=str,
                        help='Directory for saving the models. Default is saved_models/.')

    # Parse the arguments
    args = parser.parse_args()

    # Train the model
    main(args)