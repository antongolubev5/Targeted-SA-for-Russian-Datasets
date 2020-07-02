import argparse
import collections
import os
import  numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from itertools import cycle
from numpy import interp


def get_y_true(task_name):
    """
    Read file to obtain y_true answers
    """
    y_true = []
    if task_name in ["rambler2011_json_NLI_M", "rambler2011_json_QA_M"]:
        true_data_file = "data/rambler2011_json/bert-pair/test_NLI_M.tsv"

        df = pd.read_csv(true_data_file, sep='\t')
        y_true = []
        for i in range(len(df)):
            label = df['label'][i]
            assert label in ['положительно', 'отрицательно', 'нейтрально'], "error!"
            if label == 'положительно':
                n = 0
            elif label == 'отрицательно':
                n = 1
            elif label == 'нейтрально':
                n = 2
            y_true.append(n)
    elif task_name in ["rambler2011_json_NLI_B", "rambler2011_json_QA_B"]:
        true_data_file = "data/rambler2011_json/bert-pair/test_NLI_B.tsv"
        df = pd.read_csv(true_data_file, sep='\t')
        y_true = []
        for i in range(len(df)):
            label = df['label'][i]
            assert label in [0, 1], "error!"
            if label == 0:
                n = 0
            elif label == 1:
                n = 1
            y_true.append(n)
    elif task_name == "rambler2011_json_single":
        true_data_file = "data/rambler2011_json/bert-single/loc1_вообще/test.tsv"

        df = pd.read_csv(true_data_file, sep='\t')
        y_true = []
        for i in range(len(df)):
            label = df['label'][i]
            assert label in ['положительно', 'отрицательно', 'нейтрально'], "error!"
            if label == 'положительно':
                n = 0
            elif label == 'отрицательно':
                n = 1
            elif label == 'нейтрально':
                n = 2
            y_true.append(n)
    return y_true


def get_y_pred(task_name, pred_data_dir):
    """ 
    Read file to obtain y_pred and scores.
    """
    pred = []
    score = []
    if task_name in ["rambler2011_json_NLI_M", "rambler2011_json_QA_M"]:
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2]), float(s[3])])
                s = f.readline().strip().split()
    elif task_name in ["rambler2011_json_NLI_B", "rambler2011_json_QA_B"]:
        with open(pred_data_dir, "r", encoding="utf-8") as f:
            s = f.readline().strip().split()
            while s:
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2])])
                s = f.readline().strip().split()

    elif task_name == "rambler2011_json_single":
        with open(pred_data_dir, "r", encoding="utf-8") as f1_general:
            s = f1_general.readline().strip().split()
            while s:
                pred.append(int(s[0]))
                score.append([float(s[1]), float(s[2]), float(s[3])])
                s = f1_general.readline().strip().split()
    return pred, score


def get_tonal_words(task_name):
    """
        Read file to obtain tonal_word for each sample
    """
    if task_name in ["rambler2011_json_NLI_M", "rambler2011_json_QA_M"]:
        true_data_file = "data/rambler2011_json/bert-pair/test_NLI_M.tsv"
        df = pd.read_csv(true_data_file, sep='\t')
        tonal_words = list(df['tonal_word'])
    elif task_name in ["rambler2011_json_NLI_B", "rambler2011_json_QA_B"]:
        true_data_file = "data/rambler2011_json/bert-pair/test_NLI_B.tsv"
        df = pd.read_csv(true_data_file, sep='\t')
        tonal_words = list(df['tonal_word'])
    elif task_name == "rambler2011_json_single":
        true_data_file = "data/rambler2011_json/bert-single/loc1_вообще/test.tsv"
        df = pd.read_csv(true_data_file, sep='\t')
        tonal_words = list(df['tonal_word'])

    return tonal_words


def rambler2011_json_strict_acc(y_true, y_pred):
    """
    Calculate "strict Acc" of aspect detection task of Rambler2011.
    """
    total_cases = int(len(y_true))
    true_cases = 0
    for i in range(total_cases):
        if y_true[i] != y_pred[i]:
            continue
        true_cases += 1
    aspect_strict_Acc = true_cases / total_cases

    return aspect_strict_Acc


def rambler2011_json_macro_F1(y_true, y_pred):
    """
    Calculate "Macro-F1" of aspect detection task of Rambler2011.
    """
    p_all = 0
    r_all = 0
    count = 0
    for i in range(len(y_pred)):
        a = set()
        b = set()
        for j in range(1):
            if y_pred[i + j] != 0:
                a.add(j)
            if y_true[i + j] != 0:
                b.add(j)
        if len(b) == 0:
            continue
        a_b = a.intersection(b)
        if len(a_b) > 0:
            p = len(a_b) / len(a)
            r = len(a_b) / len(b)
        else:
            p = 0
            r = 0
        count += 1
        p_all += p
        r_all += r

    Ma_p = p_all / count
    Ma_r = r_all / count
    aspect_Macro_F1 = 2 * Ma_p * Ma_r / (Ma_p + Ma_r)

    return aspect_Macro_F1


def rambler2011_json_AUC_Acc(y_true, score):
    """
    Calculate "Macro-AUC" of both aspect detection and sentiment classification tasks of Rambler2011.
    Calculate "Acc" of sentiment classification task of Rambler2011.
    """
    # aspect-Macro-AUC
    aspect_y_true = []
    aspect_y_score = []
    aspect_y_trues = [[]]
    aspect_y_scores = [[]]
    for i in range(len(y_true)):
        if y_true[i] > 0:
            aspect_y_true.append(0)
        else:
            aspect_y_true.append(1)  # "None": 1
        tmp_score = score[i][0]  # probability of "None"
        aspect_y_score.append(tmp_score)
        aspect_y_trues[i].append(aspect_y_true[-1])
        aspect_y_scores[i].append(aspect_y_score[-1])

    aspect_auc = []
    for i in range(1):
        aspect_auc.append(metrics.roc_auc_score(aspect_y_trues[i], aspect_y_scores[i]))
    aspect_Macro_AUC = np.mean(aspect_auc)

    # sentiment-Macro-AUC
    sentiment_y_true = []
    sentiment_y_pred = []
    sentiment_y_score = []
    sentiment_y_trues = [[], [], [], []]
    sentiment_y_scores = [[], [], [], []]
    for i in range(len(y_true)):
        if y_true[i] > 0:
            sentiment_y_true.append(y_true[i] - 1)  # "Postive":0, "Negative":1
            tmp_score = score[i][2] / (score[i][1] + score[i][2])  # probability of "Negative"
            sentiment_y_score.append(tmp_score)
            if tmp_score > 0.5:
                sentiment_y_pred.append(1)  # "Negative": 1
            else:
                sentiment_y_pred.append(0)
            sentiment_y_trues[i].append(sentiment_y_true[-1])
            sentiment_y_scores[i].append(sentiment_y_score[-1])

    sentiment_auc = []
    for i in range(1):
        sentiment_auc.append(metrics.roc_auc_score(sentiment_y_trues[i], sentiment_y_scores[i]))
    sentiment_Macro_AUC = np.mean(sentiment_auc)

    # sentiment Acc
    sentiment_y_true = np.array(sentiment_y_true)
    sentiment_y_pred = np.array(sentiment_y_pred)
    sentiment_Acc = metrics.accuracy_score(sentiment_y_true, sentiment_y_pred)

    return aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC


def get_acc_by_words(task_name, pred_data_dir):
    """
    подсчет точности прогнозов по конкретным словам словаря
    """
    acc_words = collections.OrderedDict()
    y_true = get_y_true(task_name)
    y_pred, _ = get_y_pred(task_name, pred_data_dir)
    tonal_words = get_tonal_words(task_name)
    tonal_words_vocab = set(tonal_words)
    data = {'y_true': y_true, 'y_pred': y_pred, 'tonal_word': tonal_words}
    df = pd.DataFrame.from_dict(data)

    for tonal_word in tonal_words_vocab:
        acc_words[tonal_word] = len(df.loc[(df['tonal_word'] == tonal_word) & (df['y_true'] == df['y_pred'])]) / len(
            df[df['tonal_word'] == tonal_word])
    return acc_words


def plot_acc_by_words(acc_by_words_file):
    """
    отрисовка гистограммы точностей по конкретным словам из словаря
    """
    acc_words = dict()
    with open(acc_by_words_file, 'r') as f_acc_words:
        for line in f_acc_words:
            acc_words[line.strip().split('\t')[0]] = float(line.strip().split('\t')[1])
    acc_words = {k: v for k, v in sorted(acc_words.items(), key=lambda item: item[1], reverse=True)}

    sns.set(style="darkgrid")
    f, ax = plt.subplots(figsize=(10, 15))
    sns.set_color_codes("dark")
    fig = sns.barplot(x=[value for key, value in acc_words.items()], y=[key for key, value in acc_words.items()],
                      color='b')
    # plt.xlim([0, 100])
    ax.set(ylabel="", xlabel="Гистограмма точности по оценочным словам")
    figure = fig.get_figure()
    figure.savefig(acc_by_words_file[:-3] + "png", dpi=600, bbox_inches='tight')
    # plt.show()


def plot_roc_auc_curves(y_true, y_pred):
    """
    plot_roc_auc_curves for each class + weighted average
    """
    sentiments = {0: 'положительно', 1: 'отрицательно', 2: 'нейтрально'}
    y_true = np.array(label_binarize(y_true, classes=[0, 1, 2]))
    y_pred = np.array(label_binarize(y_pred, classes=[0, 1, 2]))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='среднее значение по всем классам (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC of class {0} (area = {1:0.2f})'''.format(sentiments[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC for Multiclass ABSA problem')
    plt.legend(loc="lower right")
    plt.savefig('roc_auc_scores.png')
    plt.show()


def own_classification_report(y_true, y_pred, task):
    """
    returns metrics for semeval2016 task (binary-task in our terms)
    average between f1pos and f1neg
    """
    confusion_matrix = {}
    if task == 'binary':
        y_true_threes = [y_true[i:i + 3] for i in range(0, len(y_true), 3)]
        y_pred_threes = [y_pred[i:i + 3] for i in range(0, len(y_pred), 3)]
        confusion_matrix = {0: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, },
                            1: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, },
                            2: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, }}
        for i in range(len(y_true_threes)):
            for j in range(3):
                if y_true_threes[i][j] == y_pred_threes[i][j] == 1:
                    confusion_matrix[j]['TP'] += 1
                elif y_true_threes[i][j] == y_pred_threes[i][j] == 0:
                    confusion_matrix[j]['TN'] += 1
                elif y_true_threes[i][j] == 1 and y_pred_threes[i][j] == 0:
                    confusion_matrix[j]['FN'] += 1
                elif y_true_threes[i][j] == 0 and y_pred_threes[i][j] == 1:
                    confusion_matrix[j]['FP'] += 1
                else:
                    pass
    elif task == 'multi':
        confusion_matrix = {0: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, },
                            1: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, },
                            2: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, }}

        for i in range(len(y_true)):
            if y_true[i] == y_pred[i] == 0:
                confusion_matrix[0]['TP'] += 1
                confusion_matrix[1]['TN'] += 1
                confusion_matrix[2]['TN'] += 1
            if y_true[i] == y_pred[i] == 1:
                confusion_matrix[0]['TN'] += 1
                confusion_matrix[1]['TP'] += 1
                confusion_matrix[2]['TN'] += 1
            if y_true[i] == y_pred[i] == 2:
                confusion_matrix[0]['TN'] += 1
                confusion_matrix[1]['TN'] += 1
                confusion_matrix[2]['TP'] += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                confusion_matrix[0]['FN'] += 1
                confusion_matrix[1]['FP'] += 1
                confusion_matrix[2]['TN'] += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                confusion_matrix[0]['FP'] += 1
                confusion_matrix[1]['FN'] += 1
                confusion_matrix[2]['TN'] += 1
            if y_true[i] == 1 and y_pred[i] == 2:
                confusion_matrix[0]['TN'] += 1
                confusion_matrix[1]['FN'] += 1
                confusion_matrix[2]['FP'] += 1
            if y_true[i] == 2 and y_pred[i] == 1:
                confusion_matrix[0]['TN'] += 1
                confusion_matrix[1]['FP'] += 1
                confusion_matrix[2]['FN'] += 1
            if y_true[i] == 0 and y_pred[i] == 2:
                confusion_matrix[0]['FN'] += 1
                confusion_matrix[1]['TN'] += 1
                confusion_matrix[2]['FP'] += 1
            if y_true[i] == 2 and y_pred[i] == 0:
                confusion_matrix[0]['FP'] += 1
                confusion_matrix[1]['TN'] += 1
                confusion_matrix[2]['FN'] += 1
    else:
        return 'error'
    # if any ==0 then add-1 smoothing
    if any([element == 0 for element in
            [item for sublist in [elem.values() for elem in [dct for dct in confusion_matrix.values()]] for item in
             sublist]]):
        for cls in confusion_matrix.values():
            for number in cls.keys():
                cls[number] += 1
    # macro
    precisions = [confusion_matrix[0]['TP'] / (confusion_matrix[0]['TP'] + confusion_matrix[0]['FP']),
                  confusion_matrix[1]['TP'] / (confusion_matrix[1]['TP'] + confusion_matrix[1]['FP']),
                  confusion_matrix[2]['TP'] / (confusion_matrix[2]['TP'] + confusion_matrix[2]['FP'])]
    recalls = [confusion_matrix[0]['TP'] / (confusion_matrix[0]['TP'] + confusion_matrix[0]['FN']),
               confusion_matrix[1]['TP'] / (confusion_matrix[1]['TP'] + confusion_matrix[1]['FN']),
               confusion_matrix[2]['TP'] / (confusion_matrix[2]['TP'] + confusion_matrix[2]['FN'])]

    f1_macro_pos_neg = np.mean([2 * precisions[0] * recalls[0] / (precisions[0] + recalls[0]),
                                2 * precisions[1] * recalls[1] / (precisions[1] + recalls[1])])

    f1_macro = np.mean([2 * precisions[0] * recalls[0] / (precisions[0] + recalls[0]),
                        2 * precisions[1] * recalls[1] / (precisions[1] + recalls[1]),
                        2 * precisions[2] * recalls[2] / (precisions[2] + recalls[2])])

    # micro
    precision = (confusion_matrix[0]['TP'] + confusion_matrix[1]['TP']) / (
            confusion_matrix[0]['TP'] + confusion_matrix[1]['TP'] + confusion_matrix[0]['FP'] + confusion_matrix[1][
        'FP'])
    recall = (confusion_matrix[0]['TP'] + confusion_matrix[1]['TP']) / (
            confusion_matrix[0]['TP'] + confusion_matrix[1]['TP'] + confusion_matrix[0]['FN'] + confusion_matrix[1][
        'FN'])

    f1_micro_pos_neg = 2 * precision * recall / (precision + recall)

    precision = (confusion_matrix[0]['TP'] + confusion_matrix[1]['TP'] + confusion_matrix[2]['TP']) / (
            confusion_matrix[0]['TP'] + confusion_matrix[1]['TP'] + confusion_matrix[2]['TP'] + confusion_matrix[0][
        'FP'] + confusion_matrix[1]['FP'] + confusion_matrix[2]['FP'])
    recall = (confusion_matrix[0]['TP'] + confusion_matrix[1]['TP'] + confusion_matrix[2]['TP']) / (
            confusion_matrix[0]['TP'] + confusion_matrix[1]['TP'] + confusion_matrix[2]['TP'] + confusion_matrix[0][
        'FN'] + confusion_matrix[1]['FN'] + confusion_matrix[2]['FN'])

    f1_micro = 2 * precision * recall / (precision + recall)

    return f1_macro_pos_neg, f1_micro_pos_neg, f1_macro, f1_micro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["rambler2011_json_single", "rambler2011_json_NLI_M", "rambler2011_json_QA_M",
                                 "rambler2011_json_NLI_B", "rambler2011_json_QA_B"],
                        help="The name of the task to evalution.")
    parser.add_argument("--pred_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pred data dir.")
    # arguments = ['--task_name=rambler2011_json_single', '--pred_data_dir=results/rambler2011_json/single/test_ep_1.txt']
    # args = parser.parse_args(arguments)
    args = parser.parse_args()

    output_log_file = os.path.join('results', 'rambler2011_json', args.task_name[17:], "metrics.txt")
    acc_by_words_file = os.path.join('results', 'rambler2011_json', args.task_name[17:], "acc_by_words.txt")
    result = collections.OrderedDict()
    acc_words = {}

    if args.task_name in ["rambler2011_json_NLI_M", "rambler2011_json_QA_M"]:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
        f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
        roc_auc = metrics.roc_auc_score(label_binarize(y_true, classes=[0, 1, 2]),
                                        label_binarize(y_pred, classes=[0, 1, 2]),
                                        average='weighted',
                                        multi_class='ovo')
        f1_macro_posneg, f1_micro_posneg, f1_macro, f1_micro = own_classification_report(y_true, y_pred, 'multi')
        result = {'Strict_Accuracy': accuracy,
                  'F1_weighted': f1_weighted,
                  'F1_macro': f1_macro,
                  'F1_micro': f1_micro,
                  'AUC_score': roc_auc,
                  'F1_macro_posneg': f1_macro_posneg,
                  'F1_micro_posneg': f1_micro_posneg,
                  }
    elif args.task_name in ["rambler2011_json_NLI_B", "rambler2011_json_QA_B"]:
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
        f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        acc_words = get_acc_by_words(args.task_name, args.pred_data_dir)
        f1_macro_posneg, f1_micro_posneg = own_classification_report(y_true, y_pred, 'binary')
        result = {'Strict_Accuracy': accuracy,
                  'F1_weighted': f1_weighted,
                  'F1_macro': f1_macro,
                  'F1_micro': f1_micro,
                  'AUC_score': roc_auc,
                  'F1_macro_posneg': f1_macro_posneg,
                  'F1_micro_posneg': f1_micro_posneg,
                  }
    elif args.task_name == "rambler2011_json_single":
        y_true = get_y_true(args.task_name)
        y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
        f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
        roc_auc = metrics.roc_auc_score(label_binarize(y_true, classes=[0, 1, 2]),
                                        label_binarize(y_pred, classes=[0, 1, 2]),
                                        average='weighted',
                                        multi_class='ovo')
        acc_words = get_acc_by_words(args.task_name, args.pred_data_dir)
        f1_macro_posneg, f1_micro_posneg, f1_macro, f1_micro = own_classification_report(y_true, y_pred, 'multi')
        result = {'Strict_Accuracy': accuracy,
                  'F1_weighted': f1_weighted,
                  'F1_macro': f1_macro,
                  'F1_micro': f1_micro,
                  'AUC_score': roc_auc,
                  'F1_macro_posneg': f1_macro_posneg,
                  'F1_micro_posneg': f1_micro_posneg,
                  }
        # plot_roc_auc_curves(y_true, y_pred)

        # for key in result.keys():
        #     print(key, "=", str(round(result[key] * 100, 2)))

    with open(output_log_file, "w") as writer:
        for key in result.keys():
            writer.write(key + "\t" + str(round(result[key] * 100, 2)) + '\n')

    with open(acc_by_words_file, "w") as writer:
        for key in acc_words.keys():
            writer.write(key + "\t" + str(round(acc_words[key] * 100, 2)) + '\n')

    # plot_acc_by_words(acc_by_words_file)


if __name__ == "__main__":
    main()
