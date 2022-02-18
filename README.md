# ir2

This repository contains research on efficient Dense Passage Retrieval (DPR). We research the effects of replacing the default BERT encoder models with smaller models (DistilBERT, TinyBERT, and ELECTRA) and the effects of using different embedding sizes. We evaluate on the retrieval performance on the Natural Questions (NQ) dataset.

## Content

This repository consists of the following scripts and folders:

- **data_download/download_nq_data**: allows you to download the Natural Questions dataset.
- **experiments/train_dpr.py**: allows you train a DPR model with an encoder model of choice.
- **experiments/evaluate_dpr.py**: allows you to evalaute the trained DPR model.
- **experiments/combined_dpr_model.py**: combines the question and passage models for joint training.
- **experiments/custom_dpr.py**: adapts the DPR code from Huggingface such that it allows for swapping the encoder models.
- **data_analysis/length_analysis.py**: calculates the tokenized lengths of the questions and passages in the train and dev datasets.
- **data_analysis/Data_analysis.ipynb**: Python Notebook containing the analysis of the tokenized lengths and our experiment results.
- **environments/**: contains the environment files for running locally or Surfsara's Lisa cluster.
- **job_files/**: contains template job files for running experiments on the Lisa cluster.
- **evaluation_outputs/**: contains the evaluation results we achieved in our experiments.

The accompanying paper of this research can be found in this repository as **Paper.pdf**.

## Pre-trained Models

Our pre-trained models can be found in [this Drive folder](https://drive.google.com/drive/folders/1N53ssc81cPul118vP6ycVgyQxnBH5TfD?usp=sharing).

## Model Naming Convention

When using the evaluation script, the models need to adhere to a specific naming convention in order to be loaded. Please make sure that your model folder names adhere to the following naming convention: **{model_name}{max*seq_length}*{embeddings_size}**. For example, when evaluating the BERT-based model with max sequence length of 256 and standard embeddings size of 768, please use **bert256_0**. Another example, when evaluating the DistilBERT-based model with max sequence length 256 and a smaller embeddings size of 512, please use **distilbert256_512**. Note that some of our models in the Drive folder contain an extra parameter at the end, this is the batch size. However, this is not detected by the evaluation script, so please remove the extra parameter from the end. For example, when evaluating **bert256_0_8** (so BERT-based model with max sequence length 256, standard embeddings size and a batch size of 8), rename it to **bert256_0**.

## Prerequisites

- Anaconda. Available at: https://www.anaconda.com/distribution/

## Getting Started

1. Open Anaconda prompt and clone this repository (or download and unpack zip):

```bash
git clone https://github.com/AndrewHarrison/ir2
```

2. Create the environment:

```bash
conda env create -f environments/environment.yml
```

Or use the Lisa environment when running on the SurfSara Lisa cluster:

```bash
conda env create -f environments/environment_lisa.yml
```

3. Activate the environment:

```bash
conda activate ir2
```

4. Move to the directory:

```bash
cd ir2
```

5. Download the Natural Questions dataset:

```bash
python data_download/download_nq_data.py --resource data.retriever.nq --output_dir data/
```

6. Run the training script for DPR with a BERT encoder:

```bash
python train_dpr.py
```

Or download one of our models from the Drive folder and evaluate:

```bash
python evaluate_dpr.py
```

## Arguments

The DPR models can be trained using the following command line arguments:

```bash
usage: train_dpr.py [-h] [--model MODEL] [--max_seq_length MAX_SEQ_LENGTH] [--embeddings_size EMBEDDINGS_SIZE]
                        [--dont_embed_title] [--data_dir DATA_DIR] [--lr LR] [--warmup_steps WARMUP_STEPS]
                        [--dropout DROPOUT] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--save_dir SAVE_DIR]
                        [--seed SEED]

optional arguments:
  -h, --help                            Show help message and exit.
  --model MODEL                         What encoder model to use. Options: ['bert', 'distilbert', 'electra', 'tinybert']. Default is 'bert'.
  --max_seq_length MAX_SEQ_LENGTH       Maximum tokenized sequence length. Default is 256.
  --embeddings_size EMBEDDINGS_SIZE     Size of the model embeddings. Default is 0 (standard model embeddings sizes).
  --dont_embed_title                    Do not embed passage titles. Titles are embedded by default.
  --data_dir DATA_DIR                   Directory where the data is stored. Default is data/downloads/data/retriever/.
  --lr LR                               Learning rate to use during training. Default is 1e-5.
  --warmup_steps WARMUP_STEPS           Number of warmup steps. Default is 100.
  --dropout DROPOUT                     Dropout rate to use during training. Default is 0.1.
  --n_epochs N_EPOCHS                   Number of epochs to train for. Default is 40.
  --batch_size BATCH_SIZE               Training batch size. Default is 16.
  --save_dir SAVE_DIR                   Directory for saving the models. Default is saved_models/.
  --seed SEED                           Seed to use during training. Default is 1234.
```

The DPR models can be evaluated using the following command line arguments:

```bash
usage: evaluate_dpr.py [-h] [--model MODEL] [--load_dir LOAD_DIR] [--max_seq_length MAX_SEQ_LENGTH]
                        [--embeddings_size EMBEDDINGS_SIZE] [--batch_size BATCH_SIZE] [--dont_embed_title]
                        [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--seed SEED]

optional arguments:
  -h, --help                            Show help message and exit.
  --model MODEL                         What encoder model to use. Options: ['bert', 'distilbert', 'electra', 'tinybert']. Default is 'bert'.
  --load_dir LOAD_DIR                   Directory for loading the trained models. Default is saved_models/.
  --max_seq_length MAX_SEQ_LENGTH       Maximum tokenized sequence length. Default is 256.
  --embeddings_size EMBEDDINGS_SIZE     Size of the model embeddings. Default is 0 (standard model embeddings sizes).
  --batch_size BATCH_SIZE               Batch size to use for encoding questions and passages. Default is 512.
  --dont_embed_title                    Do not embed passage titles. Titles are embedded by default.
  --data_dir DATA_DIR                   Directory where the data is stored. Default is data/downloads/data/retriever/.
  --output_dir OUTPUT_DIR               Directory for saving the model evaluation metrics. Default is evaluation_outputs/.
  --seed SEED                           Seed to use during training. Default is 1234.
```

## Authors

- deleted for blind submission

## Acknowledgements

- The **data_download/download_nq_data.py** file was copied from the original [DPR Github](https://github.com/facebookresearch/DPR).
