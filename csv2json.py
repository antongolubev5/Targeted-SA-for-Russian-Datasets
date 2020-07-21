import json
import os
import pandas as pd
from tqdm import tqdm
import argparse


def create_sample(directory_path, contexts, sample_type: str):
    """
    создание json-выборки (train/test)
    contexts - df с сэмплами
    contexts_file_name - имя исходного csv-файла с данными
    sample_type - train/test/val выборка
    """
    json_file = open(os.path.join(directory_path, 'dataset' + '-' + sample_type + '.json'), 'w')
    json_contexts = []

    for i in tqdm(range(len(contexts))):
        tonal_words_summary = []
        text = contexts.iloc[i]['text_tok']
        textID = str(contexts.iloc[i]['textID'])
        target_entity = contexts.iloc[i]['rus_entity']
        sentiment = contexts.iloc[i]['label']
        if sentiment == 0:
            sentiment = 'нейтрально'
        elif sentiment == 1:
            sentiment = 'положительно'
        elif sentiment == -1:
            sentiment = 'отрицательно'
        text = text.replace(target_entity, 'LOCATION1')
        tonal_words_summary.append({'sentiment': sentiment, 'aspect': 'вообще', 'target_entity': 'LOCATION1',
                                    'tonal_word': target_entity})
        json_contexts.append(
            {'opinions': tonal_words_summary,
             'id': str(i),
             'textID': textID,
             'text': text})
    json_file.write(json.dumps(json_contexts, indent=4, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True,
                        choices=["romip_2012", "sentirueval_2015_banks", "sentirueval_2015_telecom",
                                 "sentirueval_2016_banks", "sentirueval_2016_telecom"],
                        help="Dataset for evaluation.")
    args = parser.parse_args()

    train = pd.read_csv(os.path.join('preprocessed_data', args.dataset, 'train.csv'), sep='\t')
    test = pd.read_csv(os.path.join('preprocessed_data', args.dataset, 'test.csv'), sep='\t')

    for sample, name in [(test, 'test'), (test, 'dev'), (train, 'train')]:
        create_sample('data/dataset', sample, name)


if __name__ == "__main__":
    main()
