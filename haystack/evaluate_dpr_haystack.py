# imports
import os
import argparse
import json
import torch
import pandas as pd
from haystack import Document
from haystack.retriever.dense import DensePassageRetriever
from haystack.document_store import FAISSDocumentStore
from haystack.pipeline import DocumentSearchPipeline
from haystack.utils import print_documents


def load_questions_passages(json_file):
    """
    Function that converts the json file to lists of question-answer
    pairs and documents..
    Inputs:
        json_file - Path to JSON file
    Outputs:
        question_answers - List of tuples containing questions and answers
        documents - List of document items
    """

    # Open the JSON file
    question_answers = []
    documents: List[Document] = []
    with open(json_file) as f:
        data = json.load(f)
        print(json.dumps(data[:5], indent=4, sort_keys=True))
        for instance in data:
            # Create a question-answer pair for each instance
            question_answers.append((instance['question'], instance['positive_ctxs'][0]['passage_id']))
            # Add the answer to the documents list
            documents.append(
                Document(
                    text=instance['positive_ctxs'][0]['text'],
                    meta={
                        "title": instance['positive_ctxs'][0]['title'],
                        "passage_id": instance['positive_ctxs'][0]['passage_id'],
                    }
                )
            )
    
    # Return the question-answer and documents lists
    return question_answers, documents


def convert_to_documents(df):
    """
    Function that converts the pandas dataframe to list of documents.
    Inputs:
        df - Pandas DataFrame containing the documents
    Outputs:
        documents - List of document items
    """

    # Use data to initialize Document objects
    ids = list(df["id"].values)
    titles = list(df["title"].values)
    texts = list(df["text"].values)
    documents: List[Document] = []
    for id, title, text in zip(ids, titles, texts):
        documents.append(
            Document(
                text=text,
                meta={
                    "id": id,
                    "title": title,
                }
            )
        )
    
    # Return the documents
    return documents


def evaluate_model(args):
    """
    Function for evaluating the model.
    Inputs:
        args - Namespace object from the argument parser
    Outputs:
        ?
    """

    # load the passages
    print('Loading passages...')
    dev_filename = "nq-dev.json"
    question_answers, documents = load_questions_passages(args.data_dir + dev_filename)
    document_store = FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        return_embedding=True,
        use_gpu=True
    )
    document_store.delete_documents()
    document_store.write_documents(documents)
    print('Passages loaded')
    
    # # load the passages
    # print('Loading passages...')
    # passages_filename = "psgs_w100.tsv"
    # df = pd.read_csv(args.passage_dir + passages_filename, sep='\t', header=0)
    # df.fillna(value="", inplace=True)
    # documents = convert_to_documents(df)
    # document_store = FAISSDocumentStore(
    #     faiss_index_factory_str="Flat",
    #     return_embedding=True,
    #     use_gpu=True
    # )
    # document_store.delete_documents()
    # document_store.write_documents(documents)
    # print('Passages loaded')

    # load the model
    print('Loading model..')
    dpr_retriever = reloaded_retriever = DensePassageRetriever.load(
        load_dir=args.model_dir, 
        document_store=document_store,
        use_gpu=True,
    )
    document_store.update_embeddings(dpr_retriever, update_existing_embeddings=True)
    p_retrieval = DocumentSearchPipeline(dpr_retriever)
    print('Model loaded')

    # evaluate the model
    print('Starting evaluation..')
    correct_instances = 0
    for question, answer in question_answers:
        res = p_retrieval.run(
            query=question,
            params={"Retriever": {"top_k": args.top_k}},
        )
        predicted_passage = res['documents'][0]['meta']['passage_id']
        #print('Predicted: {} | Actual: {}'.format(predicted_passage, answer))
        if predicted_passage == answer:
            correct_instances += 1
    accuracy = correct_instances / len(question_answers)
    print('Accuracy: {}'.format(round(accuracy, 2)))
    print('Evaluation finished')


def main(args):
    """
    Function for handling the arguments and starting the evaluation.
    Inputs:
        args - Namespace object from the argument parser
    """

    # check if GPU is available
    check_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print the given parameters
    print('-----EVALUATION PARAMETERS-----')
    print('Device: {}'.format(check_device))
    print('Model directory: {}'.format(args.model_dir))
    print('Top k: {}'.format(args.top_k))
    print('-----------------------------')

    # start evaluation
    evaluate_model(args)


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hyperparameters
    parser.add_argument('--model_dir', default='saved_models', type=str,
                        help='Directory where the saved model is located. Default is saved_models.')
    parser.add_argument('--top_k', default=3, type=int,
                        help='Number of documents to return per query. Default is 3.')
    parser.add_argument('--data_dir', default='data/downloads/data/retriever/', type=str,
                        help='Directory where the dev data is stored. Default is data/downloads/data/retriever/.')
    parser.add_argument('--passage_dir', default='wikipedia_data/downloads/data/wikipedia_split/', type=str,
                        help='Directory where the passage data is stored. Default is wikipedia_data/downloads/data/wikipedia_split/.')

    # parse the arguments
    args = parser.parse_args()

    # train the model
    main(args)