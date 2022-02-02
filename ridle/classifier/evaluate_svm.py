import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from ridle import ROOT_DIR
from ridle.utils import aggregate_type_mappings, exclude_external_types
from ridle.datasets import fb_yago_subsets


parser = argparse.ArgumentParser(
    description='Support Vector Machine Prediction using Ridle',
)
parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
parser = parser.parse_args()

dataset = parser.dataset

dataset_names = ['FB-L1', 'FB-L2-org', 'FB-L2-person', 'FB-L3-person-artist', 'YAGO-L1', 'YAGO-L2-org',
                 'YAGO-L2-body_of_water', 'YAGO-L2-person', 'YAGO-L3-person-writer', 'YAGO-L3-person-artist',
                 'YAGO-L3-person-player', 'YAGO-L3-person-scientist']

for dataset in ['umls']:
    print(f'Training on dataset {dataset}')

    # Load Representations
    print('Reading Data...')
    df = pd.read_csv(f'{ROOT_DIR}/dataset/{dataset}/embedding.csv')

    # Load mapping
    if 'dbp' in dataset.lower():
        mapping = pd.read_json(f'{ROOT_DIR}/dataset/dbp_type_mapping.json')
    elif 'wd' in dataset.lower() or 'wikidata' in dataset.lower():
        mapping = pd.read_json(f'{ROOT_DIR}/dataset/wd_mapping_type.json')
    elif 'fb' in dataset.lower():
        fb_types = pd.read_csv(f'{ROOT_DIR}/dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
        mapping = aggregate_type_mappings(fb_types)

        if dataset in fb_yago_subsets:
            mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
    elif 'yago' in dataset.lower():
        yago_types = pd.read_csv(f'{ROOT_DIR}/dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
        yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], ['', ''], regex=True)
        mapping = aggregate_type_mappings(yago_types)

        if dataset in fb_yago_subsets:
            mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
    else:
        mapping = pd.read_json(f'{ROOT_DIR}/dataset/{dataset}/type_mapping.json')

    # merge them
    print('Processing Data...')
    r = pd.merge(df, mapping, on='S')

    K_FOLD = 10
    mlb = MultiLabelBinarizer()
    fold_no = 1
    f1_macro, f1_micro, f1_weighted = [], [], []
    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
    targets = mlb.fit_transform(r['Class'])
    inputs = r.drop(['S', 'Class'], axis=1).values

    for train, test in kfold.split(inputs, targets):
        model = SVC(kernel='rbf', verbose=True)
        model.fit(inputs[train], targets[train])

        y_pred = model.predict(inputs[test])
        accuracy = accuracy_score(targets[test], y_pred)

        f1_macro.append(f1_score(targets[test], y_pred, average='macro', zero_division=1))
        f1_micro.append(f1_score(targets[test], y_pred, average='micro', zero_division=1))
        f1_weighted.append(f1_score(targets[test], y_pred, average='weighted', zero_division=1))

        print('Score for fold', fold_no, ':', 'F1-Macro:', f1_macro[-1],
              'F1-Micro:', f1_micro[-1])

        fold_no += 1

    # Provide average scores
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print('> F1-Macro:', np.mean(f1_macro), '(+-', np.std(f1_macro), ')')
    print('> F1-Micro:', np.mean(f1_micro), '(+-', np.std(f1_micro), ')')
    print('------------------------------------------------------------------------')

    # Save results to file
    result = {}
    f1_macro = np.array(f1_macro)
    f1_micro = np.array(f1_micro)
    f1_weighted = np.array(f1_weighted)
    result['F1-Macro'] = np.mean(f1_macro)
    result['F1-Macro_std'] = np.std(f1_macro)
    result['F1-Micro'] = np.mean(f1_micro)
    result['F1-Micro_std'] = np.std(f1_micro)
    result['F1-Weighted'] = np.mean(f1_weighted)
    result['F1-Weighted_std'] = np.std(f1_weighted)
    result['Dataset'] = dataset
    result['method'] = 'Ridle'
    df_result = pd.DataFrame([result])
    print(df_result)

    if os.path.isfile(f'{ROOT_DIR}/f1_scores/evaluation_svm.csv'):
        df_result.to_csv(f'{ROOT_DIR}/f1_scores/evaluation_svm.csv', mode='a', header=False, index=False)
    else:
        df_result.to_csv(f'{ROOT_DIR}/f1_scores/evaluation_svm.csv', index=False)
