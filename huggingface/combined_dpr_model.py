# Imports
import sys
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPR(nn.Module):
    """
    DPR model class for learning question and context encoder at the same time.
    """

    def __init__(self, question_encoder, context_encoder, question_tokenizer, context_tokenizer):
        """
        Function initializing the DPR model.
        Inputs:
            question_encoder - Model that encodes the question
            context_encoder - Model that encodes the contexts
            question_tokenizer - Tokenizer for the question encoder
            context_tokenizer - Tokenizer for the context encoder
        """
        super(DPR, self).__init__()

        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer

        self.log_softmax = nn.LogSoftmax(dim=1)


    def get_context_vectors(self, context):
        """
        Retrieve the vector for the given context.
        Inputs:
            context - The context to retrieve the vector for
        """

        c_vector = self.context_encoder(
            input_ids = context.input_ids, 
            attention_mask = context.attention_mask
        )
        c_vector = c_vector.pooler_output
        return c_vector


    def get_question_vector(self, question):
        """
        Retrieve the vector for the given question.
        Inputs:
            question - The question to retrieve the vector for
        """

        q_vector = self.question_encoder(
            input_ids = question.input_ids, 
            attention_mask = question.attention_mask
        )
        q_vector = q_vector.pooler_output
        return q_vector


    def dot_product(self, q_vector, c_vector):
        """
        Calculate the dot proudct between a question and context vector.
        Inputs:
            q_vector - Vector for the question
            c_vector - Vector for the context
        """

        #q_vector = q_vector.unsqueeze(1)
        sim = torch.matmul(q_vector, torch.transpose(c_vector, -2, -1))
        return sim


    def forward(self, context_input_ids, context_attention_mask, question_input_ids, question_attention_mask):
        """
        Forward function to pass through the model.
        Inputs:
            context_input_ids - Input ids of the context
            context_attention_mask - Attention mask of the context
            question_input_ids - Input ids of the question
            question_attention_mask - Attention mask of the question
        """

        # Pass context and question through the models
        dense_context = self.context_encoder(input_ids = context_input_ids, attention_mask = context_attention_mask)
        dense_question = self.question_encoder(input_ids = question_input_ids, attention_mask = question_attention_mask)
        dense_context = dense_context['pooler_output']
        dense_question = dense_question['pooler_output']

        # Calculate the similarity score
        similarity_score = self.dot_product(dense_question, dense_context)
        similarity_score = similarity_score.squeeze(1)
        logits = self.log_softmax(similarity_score)

        # Return the similarity score
        return logits