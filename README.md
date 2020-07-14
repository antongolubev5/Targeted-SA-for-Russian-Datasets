# Improving Results on Russian Sentiment Datasets

Source code and data for the article (Anton Golubev & Natalia Loukachevitch, AINL 2020)

## Requirements

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4
* nltk
* sklearn
* pandas
* os
* tqdm
* json
* xml
* argparse

## Step 1: data preparation
All 5 datasets are already cleaned and preprocessed. If you are interested in raw data, corresponding links are provided in the article.

Run following command to prepare for tasks any dataset from following:

| Dataset       | Parameter |
| ------------- | ------------- |
| News Quotes ROMIP-2013  | romip_2012  |
| SentiRuEval-2015 Telecom  | sentirueval_2015_telecom  |
| SentiRuEval-2015 Banks  | sentirueval_2015_banks  |
| SentiRuEval-2016 Telecom  | sentirueval_2016_telecom  |
| SentiRuEval-2016 Banks  | sentirueval_2016_banks  |


```
python csv2json.py --dataset parameter
```

## Step 2: prepare BERT-pytorch-model

Download [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert) and then convert a tensorflow checkpoint to a pytorch model.

For example:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```

## Step 3: train

For example, **BERT-pair-NLI_M** task on **SentiHood** dataset:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_TABSA.py \
--task_name sentihood_NLI_M \
--data_dir data/sentihood/bert-pair/ \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--output_dir results/sentihood/NLI_M \
--seed 42
```

Note:

* For SentiHood, `--task_name` must be chosen in `sentihood_NLI_M`, `sentihood_QA_M`, `sentihood_NLI_B`, `sentihood_QA_B` and `sentihood_single`. And for `sentihood_single` task, 8 different tasks (use datasets generated in step 1, see directory `data/sentihood/bert-single`) should be trained separately and then evaluated together.
* For SemEval-2014, `--task_name` must be chosen in `semeval_NLI_M`, `semeval_QA_M`, `semeval_NLI_B`, `semeval_QA_B` and `semeval_single`. And for `semeval_single` task, 5 different tasks (use datasets generated in step 1, see directory : `data/semeval2014/bert-single`) should be trained separately and then evaluated together.

## Step 4: evaluation

Evaluate the results on test set (calculate Acc, F1, etc.).

For example, **BERT-pair-NLI_M** task on **SentiHood** dataset:

```
python evaluation.py --task_name sentihood_NLI_M --pred_data_dir results/sentihood/NLI_M/test_ep_4.txt
```

