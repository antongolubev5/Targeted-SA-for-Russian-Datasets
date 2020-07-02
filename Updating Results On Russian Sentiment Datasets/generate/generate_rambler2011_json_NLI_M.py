import os

from data_utils_sentihood import *

data_dir = '../data/rambler2011_json/'
aspect2idx = {'вообще': 0}

(train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task_rambler2011_json(data_dir,
                                                                                                       aspect2idx)
# сортировка по номеру сущности + индекс предложения
train.sort(key=lambda x: x[2] + str(x[0]) + x[3][0])
val.sort(key=lambda x: x[2] + str(x[0]) + x[3][0])
test.sort(key=lambda x: x[2] + str(x[0]) + x[3][0])

dir_path = data_dir + 'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# преобразуем загруженные текстовые данные из json/xml к парам предложений для дальнейшей подачи в BERT.
# NLI_M == предложение + краткий вывод из него для каждого target (1/2) и aspect (4)
# одно предложение с маскированным loc1 трансформируем в 4 пары предложение-предложение для каждого aspect

# "LOCATION1 is in Greater London  and is a very safe place"    Positive    safety  LOCATION1
# ------------>
# "LOCATION1 is in Greater London  and is a very safe place"    location - 1 - general          None
# "LOCATION1 is in Greater London  and is a very safe place"    location - 1 - safety           Positive
# "LOCATION1 is in Greater London  and is a very safe place"    location - 1 - transit-location None
# "LOCATION1 is in Greater London  and is a very safe place"    location - 1 - price            None

print("\nNLI_M:")

for part, sample in zip(["train", "dev", "test"], [train, val, test]):
    with open(dir_path + part + "_" + "NLI_M.tsv", "w", encoding="utf-8") as f:
        f.write("id\tsentence1\tsentence2\tlabel\ttonal_word\ttextID\n")
        cnt = 0
        for v in sample:
            f.write(str(v[0]) + "\t")
            word = v[1][0].lower()
            if word == 'location1':
                f.write('location - 1')
            elif word == 'location2':
                f.write('location - 2')
            elif word[0] == '\'':
                f.write("\' " + word[1:])
            else:
                f.write(word)

            for i in range(1, len(v[1])):
                word = v[1][i].lower()
                f.write(" ")
                if word == 'location1':
                    f.write('location - 1')
                elif word == 'location2':
                    f.write('location - 2')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            if v[2] == 'LOCATION1':
                f.write('location - 1 - ')
            if v[2] == 'LOCATION2':
                f.write('location - 2 - ')
            if len(v[3]) == 1:
                f.write(v[3][0] + "\t")
            else:
                f.write("transit location\t")
            f.write(v[4] + "\t")
            f.write(v[5] + "\t")
            f.write(v[6] + "\n")
            cnt += 1
        print(('len({}) = ' + str(cnt)).format(part))

