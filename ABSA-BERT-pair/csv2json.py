import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def vocab_from_file(directory_path, file_names):
    vocab = dict()
    for file_name in file_names:
        with open(os.path.join(directory_path, file_name), 'r') as f:
            for line in f:
                vocab[line.strip().split(', ')[0]] = line.strip().split(', ')[3]
    return vocab


def check_sentiments(words, entities_vocab):
    """
    классификация предложения на группы:
    """
    neg_words = set([key for key, value in entities_vocab.items() if value == 'negative'])
    pos_words = set([key for key, value in entities_vocab.items() if value == 'positive'])
    words = set(words)
    if len(words.intersection(pos_words)) == 2:
        return 'pospos'
    elif len(words.intersection(neg_words)) == 2:
        return 'negneg'
    elif len(words.intersection(neg_words)) == len(words.intersection(pos_words)) == 1:
        return 'posneg'


def create_sample(directory_path, contexts, sample_type: str):
    """
    создание json-выборки (train/test)
    contexts - df с сэмплами
    contexts_file_name - имя исходного csv-файла с данными
    sample_type - train/test/val выборка
    """
    entities_vocab = vocab_from_file(directory_path, ['nouns_person_neg', 'nouns_person_pos'])
    json_file = open(os.path.join(directory_path, 'rambler2011' + '-' + sample_type + '.json'), 'w')
    json_contexts = []

    for i in tqdm(range(len(contexts))):
        sentiment = 0
        tonal_words_summary = []
        if contexts.iloc[i]['label'] == 0:
            tonal_words = contexts.iloc[i]['tonal_word'].split('----MY ARTIFICIAL DELIMITER----')
        else:
            tonal_words = contexts.iloc[i]['tonal_word'].split()
        text = contexts.iloc[i]['text_tok']
        for word_num in range(len(tonal_words)):
            text = text.replace(tonal_words[word_num], 'LOCATION' + str(word_num + 1))
            word2mask = tonal_words[word_num]
            target_entity = 'LOCATION' + str(word_num + 1)
            if word2mask not in entities_vocab.keys():
                sentiment = 'нейтрально'
            elif entities_vocab[word2mask] == 'positive':
                sentiment = 'положительно'
            elif entities_vocab[word2mask] == 'negative':
                sentiment = 'отрицательно'
            tonal_words_summary.append({'sentiment': sentiment, 'aspect': 'вообще', 'target_entity': target_entity,
                                        'tonal_word': word2mask})
        json_contexts.append(
            {'opinions': tonal_words_summary,
             'id': str(i),
             'text': text})
    json_file.write(json.dumps(json_contexts, indent=4, ensure_ascii=False))


def create_sample_banks(directory_path, contexts, sample_type: str):
    """
    создание json-выборки (train/test)
    contexts - df с сэмплами
    contexts_file_name - имя исходного csv-файла с данными
    sample_type - train/test/val выборка
    """
    json_file = open(os.path.join(directory_path, 'rambler2011' + '-' + sample_type + '.json'), 'w')
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
    directory_path = '/home/anton/PycharmProjects/ABSA-BERT-pair'
    train = pd.read_csv(os.path.join(directory_path, 'tkk_train_2016_cleaned.csv'), sep='\t')
    test = pd.read_csv(os.path.join(directory_path, 'tkk_test_etalon_cleaned.csv'), sep='\t')

    create_sample_banks(directory_path, test, 'test')
    create_sample_banks(directory_path, test, 'dev')
    create_sample_banks(directory_path, train, 'train')


if __name__ == "__main__":
    main()
