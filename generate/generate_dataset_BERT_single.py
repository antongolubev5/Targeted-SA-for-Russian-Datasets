import os
import pandas as pd
from data_utils_sentihood import *

data_dir = '../data/dataset/'
aspect2idx = {'вообще': 0}

(train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task_dataset(data_dir, aspect2idx)
# сортировка по номеру сущности + индекс предложения
train.sort(key=lambda x: x[2] + str(x[0]) + x[3][0])
val.sort(key=lambda x: x[2] + str(x[0]) + x[3][0])
test.sort(key=lambda x: x[2] + str(x[0]) + x[3][0])

location_name = ['loc1', 'loc2']
aspect_name = ['вообще']
dir_path = [data_dir + 'bert-single/' + i + '_' + j + '/' for i in location_name for j in aspect_name]

for path in dir_path:
    if not os.path.exists(path):
        os.makedirs(path)

print("\nsingle:")

for part in ["train", "dev", "test"]:
    count = 0
    cnt = 0
    with open(dir_path[0] + part + ".tsv", "w", encoding="utf-8") as f1_general, \
            open(dir_path[1] + part + ".tsv", "w", encoding="utf-8") as f2_general, \
            open(data_dir + "bert-pair/" + part + "_NLI_M.tsv", "r", encoding="utf-8") as f:
        df = pd.read_csv(data_dir + "bert-pair/" + part + "_NLI_M.tsv", sep='\t')
        # len_loc1 = df[df['sentence2'] == 'location - 2 - вообще'].index[0]
        len_loc1 = 100000
        s = f.readline().strip()
        s = f.readline().strip()
        f1_general.write('id' + '\t' + 'sentence' + '\t' + 'label' + "\t" + 'tonal_word' + "\t" + 'textID' + '\n')
        while s:
            count += 1
            tmp = s.split("\t")
            line = tmp[0] + "\t" + tmp[1] + "\t" + tmp[3] + "\t" + tmp[4] + "\t" + tmp[5] + "\n"
            if count <= len_loc1:  # число-граница между loc1 и loc2 в данных другой задачи
                # if count % 4 == 1: # если 4 аспекта, то сортируем по разным файлам через mod 4
                f1_general.write(line)
            else:
                f2_general.write(line)
            s = f.readline().strip()
            cnt += 1
    print(('len({}) = ' + str(cnt)).format(part))

print("\nFinished!")
