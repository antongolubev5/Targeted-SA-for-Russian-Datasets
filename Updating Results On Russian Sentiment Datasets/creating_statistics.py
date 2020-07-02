import pandas as pd
import numpy as np
import os
from collections import Counter


def creating_stats():
    directory = '/home/anton/data/ABSA'
    for dir in os.listdir(path=directory):
        if dir in ['romip2012']:
            sentences = pd.read_csv(os.path.join(directory, dir, 'all_answers.csv'), sep='\t')
            sentences['vector'] = sentences.apply(
                lambda row: [row[element] for element in sentences.columns[4:]].count(row['label']), axis=1)
            difficult = sentences[sentences['vector'] == 1]
            print(dir)
            print(difficult['rus_entity'].value_counts())
            difficult.to_csv(os.path.join(directory, dir, 'one_model_right.csv'))


def extracting_multitonal_sents():
    directory = '/home/anton/data/ABSA'
    for dir in os.listdir(path=directory):
        if dir not in ['romip2012', 'txt', 'csv']:
            sentences = pd.read_csv(os.path.join(directory, dir, 'all_answers.csv'), sep='\t')
            sents = list(sentences['text'])
            cntr = Counter(sents)
            lst = [element for element in cntr.keys() if cntr[element] == 2]
            multi_entity_sents = sentences[sentences['text'].isin(lst)]
            for i in range(len(lst)):
                if len(np.unique(multi_entity_sents[multi_entity_sents['text'] == lst[i]]['label'].values)) == 1:
                    multi_entity_sents = multi_entity_sents[multi_entity_sents['text'] != lst[i]]

            multi_entity_sents.sort_values(by=['text'], inplace=True)
            multi_entity_sents.to_csv(os.path.join(directory, dir, 'multi_polarity.csv'), index=False, sep='\t')


def creating_stats_multitonal_sents():
    directory = '/home/anton/data/ABSA'
    methods = {'NLI_M': [0, 0], 'QA_M': [0, 0], 'single': [0, 0], 'NLI_M (C)': [0, 0], 'QA_M (C)': [0, 0],
               'single (C)': [0, 0]}
    for dir in os.listdir(path=directory):
        if dir not in ['romip2012', 'txt', 'csv']:
            sentences = pd.read_csv(os.path.join(directory, dir, 'multi_polarity.csv'), sep='\t')
            for text in sentences['text'].unique():
                text_df = sentences[sentences['text'] == text]
                for method in text_df.columns[4:]:
                    if (text_df['label'].values == text_df[method].values).all():
                        methods[method][0] += 1

            for key in methods.keys():
                methods[key][1] = methods[key][0] / len(sentences) * 2
                methods[key][1] *= 100

            with open(os.path.join(directory, dir, 'methods_results.txt'), 'w') as f_methods_results:
                for key in methods.keys():
                    line = key + ' ' + str(methods[key][0]) + ' ' + str(methods[key][1]) + '\n'
                    f_methods_results.write(line)

            for key in methods.keys():
                methods[key][0] = 0
                methods[key][1] = 0


def creating_stats_by_models():
    directory = '/home/anton/data/ABSA'
    methods = {'NLI_M': [0, 0], 'QA_M': [0, 0], 'single': [0, 0], 'NLI_M (C)': [0, 0], 'QA_M (C)': [0, 0],
               'single (C)': [0, 0]}

    for dir in os.listdir(path=directory):
        if dir not in ['romip2012', 'txt', 'csv']:
            sentences = pd.read_csv(os.path.join(directory, dir, 'two_models_right.csv'), sep=',')
            for i in range(len(sentences)):
                for method in sentences.columns[5:11]:
                    if sentences.iloc[i][method] == sentences.iloc[i]['label']:
                        methods[method][0] += 1

            for key in methods.keys():
                methods[key][1] = methods[key][0] / len(sentences)
                methods[key][1] *= 100

            with open(os.path.join(directory, dir, 'two_models_right_results.txt'), 'w') as f_methods_results:
                for key in methods.keys():
                    line = key + ' ' + str(methods[key][0]) + ' ' + str(methods[key][1]) + '\n'
                    f_methods_results.write(line)

            for key in methods.keys():
                methods[key][0] = 0
                methods[key][1] = 0


def main():
    creating_stats_multitonal_sents()


if __name__ == "__main__":
    main()
