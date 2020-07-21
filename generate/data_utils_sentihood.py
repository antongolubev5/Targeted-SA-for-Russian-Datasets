# Reference: https://github.com/liufly/delayed-memory-update-entnet

from __future__ import absolute_import

import json
import operator
import os
import re
import sys
import xml.etree.ElementTree

import nltk
import numpy as np


def load_task(data_dir, aspect2idx):
    """
    загрузка и предобработка исходных данных. На выходе для каждой части выборки кортеж вида
    ([sent_id, [токенизированный текст], маска loc1/loc2, [aspect], polarity], np.array(подряд номера аспектов (0123))
    """
    in_file = os.path.join(data_dir, 'sentihood-train.json')
    train = parse_sentihood_json(in_file)

    in_file = os.path.join(data_dir, 'sentihood-dev.json')
    dev = parse_sentihood_json(in_file)

    in_file = os.path.join(data_dir, 'sentihood-test.json')
    test = parse_sentihood_json(in_file)

    train = convert_input(train, aspect2idx)
    train_aspect_idx = get_aspect_idx(train, aspect2idx)
    train = tokenize(train)

    dev = convert_input(dev, aspect2idx)
    dev_aspect_idx = get_aspect_idx(dev, aspect2idx)
    dev = tokenize(dev)

    test = convert_input(test, aspect2idx)
    test_aspect_idx = get_aspect_idx(test, aspect2idx)
    test = tokenize(test)

    return (train, train_aspect_idx), (dev, dev_aspect_idx), (test, test_aspect_idx)


def load_task_dataset(data_dir, aspect2idx):
    """
    загрузка и предобработка исходных данных. На выходе для каждой части выборки кортеж вида
    ([sent_id, [токенизированный текст], маска loc1/loc2, [aspect], polarity], np.array(подряд номера аспектов (0123))
    """
    in_file = os.path.join(data_dir, 'dataset-train.json')
    train = parse_sentihood_json(in_file)

    in_file = os.path.join(data_dir, 'dataset-dev.json')
    dev = parse_sentihood_json(in_file)

    in_file = os.path.join(data_dir, 'dataset-test.json')
    test = parse_sentihood_json(in_file)

    train = convert_input(train, aspect2idx)
    train_aspect_idx = get_aspect_idx(train, aspect2idx)
    train = tokenize(train)

    dev = convert_input(dev, aspect2idx)
    dev_aspect_idx = get_aspect_idx(dev, aspect2idx)
    dev = tokenize(dev)

    test = convert_input(test, aspect2idx)
    test_aspect_idx = get_aspect_idx(test, aspect2idx)
    test = tokenize(test)

    return (train, train_aspect_idx), (dev, dev_aspect_idx), (test, test_aspect_idx)


def get_aspect_idx(data, aspect2idx):
    """
    для выборки, полученной после convert_input создаем вектор числовых меток aspects по словарю aspect2idx
    """
    ret = []
    for _, _, _, aspect, _, _, _ in data:
        ret.append(aspect2idx[aspect])
    assert len(data) == len(ret)
    return np.array(ret)


def parse_sentihood_json(in_file):
    """
    парсинг json-файла
    на выходе список кортежей (id, text, список target-aspect тональностей [кортежи (entity, aspect, polarity)])
    если у entity нет тональности, то последний параметр = []
    """
    with open(in_file) as f:
        data = json.load(f)
    ret = []
    for d in data:
        text = d['text']
        sent_id = d['id']
        textID = d['textID']
        opinions = []
        targets = set()
        for opinion in d['opinions']:
            sentiment = opinion['sentiment']
            aspect = opinion['aspect']
            target_entity = opinion['target_entity']
            tonal_word = opinion['tonal_word']
            targets.add(target_entity)
            opinions.append((target_entity, aspect, sentiment, tonal_word))
        ret.append((sent_id, text, opinions, textID))
    return ret


def convert_input(data, all_aspects):
    """
    преобразуем данные, полученные после парсинга json'a (parse_sentihood_json) к следующему виду:
    нужно разложить все имеющиеся targets  в предложении по всем aspects
    пустым aspects присваиваем тональность=None, результат: список из кортежей
    (sent_id, текст, target, aspect, sentiment)
    """
    ret = []
    for sent_id, text, opinions, textID in data:
        for target_entity, aspect, sentiment, tonal_word in opinions:
            if aspect not in all_aspects:
                continue
            ret.append((sent_id, text, target_entity, aspect, sentiment, tonal_word, textID))
        # print(text)
        assert 'LOCATION1' in text
        targets = set(['LOCATION1'])
        if 'LOCATION2' in text:
            targets.add('LOCATION2')
        for target in targets:
            aspects = set([a for t, a, _, _ in opinions if t == target])  # ???
            none_aspects = [a for a in all_aspects if a not in aspects]
            for aspect in none_aspects:
                ret.append((sent_id, text, target, aspect, 'None', textID))
    return ret


def tokenize(data):
    """
    токенизация текста из поля text для каждой пары target-aspect +
    aspect -> [aspect]
    """
    ret = []
    for sent_id, text, target_entity, aspect, sentiment, tonal_word, textID in data:
        new_text = nltk.word_tokenize(text)
        new_aspect = aspect.split('-')
        ret.append((sent_id, new_text, target_entity, new_aspect, sentiment, tonal_word, textID))
    return ret
