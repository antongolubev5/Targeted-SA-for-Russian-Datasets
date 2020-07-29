# Improving Results on Russian Sentiment Datasets

This repository provides a source code and data for the paper "Improving Results on Russian Sentiment Datasets" (Anton Golubev and Natalia Loukachevitch, AINL 2020)

## Requirements

* python: 3.7.1
* pytorch: 1.0.0
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
All 5 datasets are already cleaned and preprocessed. If you are interested in raw data, corresponding links are provided in the paper.


| Dataset       | Name |
| ------------- | ------------- |
| News Quotes ROMIP-2013  | romip_2012  |
| SentiRuEval-2015 Telecom  | sentirueval_2015_telecom  |
| SentiRuEval-2015 Banks  | sentirueval_2015_banks  |
| SentiRuEval-2016 Telecom  | sentirueval_2016_telecom  |
| SentiRuEval-2016 Banks  | sentirueval_2016_banks  |

Run following command to prepare for tasks any dataset from table:


```
python csv2json.py --dataset name
```

## Step 2: creating samples
Now you need to create samples by converting data to a special format for submission to BERT model.

Run following commands to create samples for tasks:

```
cd generate/
bash make.sh rambler2011_json
```

## Step 3: prepare BERT-pytorch-model

Download [RuBERT or Conversational RuBERT (DeepPavlov's pre-trained models)](http://docs.deeppavlov.ai/en/master/features/models/bert.html) and then convert a tensorflow checkpoint to a pytorch model.

By the example of Conversational RuBERT:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path ru_conversational_cased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file ru_conversational_cased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path ru_conversational_cased_L-12_H-768_A-12/pytorch_model.bin
```

## Step 4: training and evaluation

Model training and testing are combined into one python script with the ability to test several methods. For example:

```
python evaluate_all_problems.py \
--bert_model ru_conversational_cased_L-12_H-768_A-12 \
--train_batch_size 12 \
--num_train_epochs 2.0 \
--train_models True \
--tasks QA_M NLI_M single QA_B NLI_B
```

A table with chosen methods and all necessary metrics will be displayed after evaluation. 

## References

```
@misc{golubev2020improving,
    title={Improving Results on Russian Sentiment Datasets},
    author={Anton Golubev and Natalia Loukachevitch},
    year={2020},
    eprint={2007.14310},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
