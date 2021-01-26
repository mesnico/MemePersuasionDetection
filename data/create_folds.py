import os
import json
import random

if __name__ == '__main__':
    how_many_folds = 6
    use_fold_zero_from_old = False
    use_dev_set = True

    label_file = os.path.join('training_set_task3', 'training_set_task3.txt')
    with open(label_file, 'r', encoding='utf8') as f:
        targets = json.load(f)

    if use_dev_set:
        label_file_dev = os.path.join('dev_set_task3_labeled', 'dev_set_task3_labeled.txt')
        with open(label_file_dev, 'r', encoding='utf8') as f:
            targets_dev = json.load(f)
            targets.extend(targets_dev)

    if use_fold_zero_from_old:
        with open('folds_old.json', 'r') as f:
            old_fold_zero = json.load(f)['0']
        ids = [x['id'] for x in targets if x['id'] not in old_fold_zero]
    else:
        ids = [x['id'] for x in targets]
    random.shuffle(ids)

    how_many_in_a_fold = int(len(ids) / (how_many_folds - 1)) if use_fold_zero_from_old else int(len(ids) / how_many_folds)

    folds = {}
    if use_fold_zero_from_old:
        folds[0] = old_fold_zero

    shift = 1 if use_fold_zero_from_old else 0
    for i in range(how_many_folds - shift):
        start = i * how_many_in_a_fold
        end = (i+1) * how_many_in_a_fold
        folds[shift+i] = ids[start:end]
    print(folds)

    # dump on file
    out_file = 'folds.json'
    with open(out_file, 'w') as f:
        json.dump(folds, f)
    print('DONE')

