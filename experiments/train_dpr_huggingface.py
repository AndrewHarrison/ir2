# Imports
import sys
import os
import argparse
import json
import random
from time import process_time
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
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
    'electra': ('google/electra-small-discriminator', ElectraTokenizer),
    'tinybert': ('huawei-noah/TinyBERT_General_4L_312D', BertTokenizer)
}


def preprocess_dataset(dataset_instance, question_tokenizer, context_tokenizer, max_seq_length):
    """
    Function for tokenizing the dataset, used with Huggingface map function.
    Inputs:
        dataset_instance - Instance from the Huggingface dataset
        question_tokenizer - Tokenizer instance to use for encoding the questions
        context_tokenizer - Tokenizer instance to use for encoding the contexts
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

    # Encode the gold context
    gold_inputs = context_tokenizer(
        dataset_instance['gold_context'],
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt',
    )
    gold_ids = gold_inputs['input_ids'].squeeze()
    gold_mask = gold_inputs['attention_mask'].squeeze()

    # Encode the negative context
    neg_inputs = context_tokenizer(
        dataset_instance['neg_context'],
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
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


def perform_training_epoch(dpr_model, device, dataloader, optimizer, scheduler, criterion, num_questions, scaler):
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
        scaler - GradScaler instance for mixed precision
    Outputs:
        epoch_loss - Average loss over the epoch
        time_elapsed - Time it took for the epoch to run
    """

    epoch_loss = 0
    epoch_correct = 0

    # Keep track of the time it takes to perform an epoch
    time_start = process_time() 

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

        with amp.autocast():
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

            # Calculate the loss
            loss = criterion(pred, true)

            # Calculate the number of correct labels
            max_score, max_ids = torch.max(pred, 1)
            correct_preds = (
                (max_ids == true).sum().cpu().detach().numpy().item()
            )
            epoch_correct += correct_preds

        # Backwards pass
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()

    # Take a step with the learning rate scheduler    
    scheduler.step()
    
    # Calculate the time it takes to do an epoch
    time_stop = process_time()
    time_elapsed = time_stop - time_start
    
    # Return the average epoch loss, epoch accuracy and elapsed time
    return epoch_loss / num_questions, epoch_correct / num_questions, time_elapsed


def train_model(args, device):
    """
    Function for training the model.
    Inputs:
        args - Namespace object from the argument parser
        device - PyTorch device to train on
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
    criterion = nn.NLLLoss(reduction="mean").to(device)
    optimizer = torch.optim.AdamW(dpr_model.parameters(), lr=args.lr, eps=1e-8)

    # Encode the training data
    print('Encoding training data..')
    train_dataset = train_dataset.map(
        lambda x: preprocess_dataset(
            x,
            question_tokenizer = question_tokenizer,
            context_tokenizer = context_tokenizer,
            max_seq_length = args.max_seq_length,
        ),
        batched=False,
    )
    train_dataset.set_format(type='pt', columns=['question_ids', 'question_mask', 'gold_context_ids', 'gold_context_mask', 'neg_context_ids', 'neg_context_mask'])
    len_dataset = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print('Training data encoded')

    # Create the linear learning rate scheduler
    def lr_lambda(current_epoch):
        return max(
            1e-7,
            float(args.n_epochs - current_epoch) / float(max(1, args.n_epochs)),
        )
    scheduler = LambdaLR(optimizer, lr_lambda, -1)

    # Create the GradScaler for mixed precision
    scaler = amp.GradScaler()

    # Train the model
    total_training_time = 0
    print('Starting training..')
    for epoch in range(1, args.n_epochs + 1):
        average_epoch_loss, epoch_accuracy, epoch_time = perform_training_epoch(
            dpr_model = dpr_model,
            device = device,
            dataloader = train_loader, 
            optimizer = optimizer,
            scheduler = scheduler, 
            criterion = criterion,
            num_questions = len_dataset,
            scaler = scaler,
        )
        total_training_time += epoch_time
        # Report on the loss
        print('Epoch : {}   Loss : {}   Accuracy: {}    Elapsed time : {}'.format(epoch, average_epoch_loss, epoch_accuracy, str(datetime.timedelta(seconds=epoch_time))))
    print('Training finished. Total training time: {}'.format(str(datetime.timedelta(seconds=total_training_time))))

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

    # Set a seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the given parameters
    print('-----TRAINING PARAMETERS-----')
    print('Device: {}'.format(device))
    print('Model: {}'.format(args.model))
    print('Maximum sequence length: {}'.format(args.max_seq_length))
    print('Learning rate: {}'.format(args.lr))
    print('Dropout rate: {}'.format(args.dropout))
    print('Num training epochs: {}'.format(args.n_epochs))
    print('Batch size: {}'.format(args.batch_size))
    print('Model saving directory: {}'.format(args.save_dir))
    print('Embed title: {}'.format(not args.dont_embed_title))
    print('Seed: {}'.format(args.seed))
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
                        choices=['bert', 'distilbert', 'electra','tinybert'])
    parser.add_argument('--max_seq_length', default=256, type=int,
                        help='Maximum sequence length. Default is 256.')
    
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
    parser.add_argument('--n_epochs', default=4, type=int,
                        help='Number of epochs to train for. Default is 4.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Training batch size. Default is 8.')
    parser.add_argument('--save_dir', default='saved_models/', type=str,
                        help='Directory for saving the models. Default is saved_models/.')
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed to use during training. Default is 1234.')

    # Parse the arguments
    args = parser.parse_args()

    # Train the model
    main(args)
