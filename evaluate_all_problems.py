import pandas as pd
import os
from tabulate import tabulate
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model",
                        default='ru_conversational_cased_L-12_H-768_A-12_pt',
                        type=str,
                        help="Pretrained BERT model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_models",
                        default=False,
                        type=bool,
                        help="Should model be trained")
    parser.add_argument("--tasks",
                        default='single QA_B NLI_B QA_M NLI_M',
                        type=str,
                        help="Tasks to evaluate")

    # arguments = ['--train_batch_size=12',
    #              '--num_train_epochs=4',
    #              '--tasks=single NLI_M QA_M']
    # args = parser.parse_args(arguments)
    args = parser.parse_args()

    folders = {'single': 'single/loc1_вообще', 'QA_M': 'pair', 'NLI_M': 'pair', 'QA_B': 'pair', 'NLI_B': 'pair'}
    info = [(task, folders[task]) for task in args.tasks.split()]
    results_all_tasks = pd.DataFrame(
        columns=['task', 'accuracy', 'f1_weighted', 'f1_macro', 'f1_micro', 'roc_auc', 'f1_macro_posneg',
                 'f1_micro_posneg'])

    for task, data_dir in tqdm(info):
        run_classifier_TABSA_parameters = \
            ['python',
             'run_classifier_TABSA.py',
             '--task_name dataset_' + task,
             '--data_dir data/dataset/bert-' + data_dir,
             '--vocab_file ' + args.bert_model + '/vocab.txt',
             '--bert_config_file ' + args.bert_model + '/bert_config.json',
             '--init_checkpoint=' + args.bert_model + '/pytorch_model.bin',
             '--eval_test',
             '--do_lower_case',
             '--max_seq_length 512',
             '--train_batch_size ' + str(args.train_batch_size),
             '--learning_rate 2e-5',
             '--num_train_epochs ' + str(args.num_train_epochs),
             '--output_dir results/dataset/' + task,
             '--seed 42']
        if args.train_models:
            os.system(' '.join(run_classifier_TABSA_parameters))

        results_df = pd.read_csv(os.path.join('results/dataset', task, 'log.txt'), sep='\t')
        best_epoch_id = results_df['test_loss'].argmax()

        evaluate_parameters = \
            ['python',
             'evaluation.py',
             '--task_name dataset_' + task,
             '--pred_data_dir results/dataset/' + task + '/test_ep_' + str(best_epoch_id + 1) + '.txt']
        os.system(' '.join(evaluate_parameters))

        results_task = dict()
        with open(os.path.join('results/dataset', task, 'metrics.txt')) as f_results:
            for line in f_results:
                results_task[line.strip().split('\t')[0]] = line.strip().split('\t')[1]

        results_all_tasks = results_all_tasks.append(pd.Series(
            [task, results_task['Strict_Accuracy'], results_task['F1_weighted'], results_task['F1_macro'],
             results_task['F1_micro'], results_task['AUC_score'], results_task['F1_macro_posneg'],
             results_task['F1_micro_posneg']],
            index=results_all_tasks.columns), ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(tabulate(results_all_tasks, headers='keys', tablefmt='psql'))


if __name__ == "__main__":
    main()
