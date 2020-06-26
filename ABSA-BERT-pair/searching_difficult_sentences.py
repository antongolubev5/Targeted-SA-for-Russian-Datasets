import pandas as pd
import os


def main():
    directory = '/home/anton/PycharmProjects/ABSA-BERT-pair/'
    sentences = pd.read_csv('tkk_test_etalon_cleaned.csv', sep='\t')[['text_tok', 'textID', 'label', 'rus_entity']]
    encoder = {-1: 1, 0: 2, 1: 0}
    sentences['label'] = sentences['label'].map(encoder)

    datasets = {'NLI_M': '/home/anton/PycharmProjects/ABSA-BERT-pair/data/rambler2011_json/bert-pair/test_NLI_M.tsv',
                'QA_M': '/home/anton/PycharmProjects/ABSA-BERT-pair/data/rambler2011_json/bert-pair/test_QA_M.tsv',
                'single': '/home/anton/PycharmProjects/ABSA-BERT-pair/data/rambler2011_json/bert-single/loc1_'
                          'вообще/test.tsv'}
    # sentences = pd.read_csv('difficult_sentences.csv', sep='\t')
    for task in ['NLI_M', 'QA_M', 'single']:
        test_data = pd.read_csv(datasets[task], sep='\t')
        lines = pd.read_csv(os.path.join(directory, 'results/rambler2011_json', task, 'log.txt'), sep='\t')
        best_epoch_id = lines['test_loss'].argmax() + 1
        results = []
        with open(os.path.join(directory, 'results/rambler2011_json', task,
                               'test_ep_' + str(best_epoch_id) + '.txt')) as f_results:
            for line in f_results.readlines():
                results.append(int(line.split()[0]))
        test_data['answer'] = results
        test_data = test_data[['textID', 'answer']]
        # test_data = test_data.rename(columns={'textID': 'textID', 'answer': task + ' (C)'})
        test_data = test_data.rename(columns={'textID': 'textID', 'answer': task})
        sentences = sentences.set_index('textID').join(test_data.set_index('textID')).reset_index()

    # sentences.drop(['textID'], axis=1, inplace=True)
    # sentences = sentences[
    #     (sentences['NLI_M (C)'] != sentences['label']) & (sentences['QA_M (C)'] != sentences['label']) & (
    #             sentences['single (C)'] != sentences['label']) & (
    #             sentences['NLI_M'] != sentences['label']) & (sentences['QA_M'] != sentences['label']) & (
    #             sentences['single'] != sentences['label'])]
    sentences.to_csv('difficult_sentences.csv', sep='\t', index=False)


if __name__ == "__main__":
    main()
