import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score


parser = argparse.ArgumentParser(
    description='Instance Type Prediction using Ridle',
)
parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
parser = parser.parse_args()

for dataset in ['Books_DBpedia', 'ChemicalCompounds_DBpedia', 'Company_DBpedia',
                'dblp', 'DBp_2016-04', 'Person_DBpedia', 'Songs_DBpedia', 'umls',
                'Universities_DBpedia']:
    print(f'Training on dataset {dataset}')

    # Load Representations
    print('Reading Data...')
    df = pd.read_csv('./dataset/{}/embedding.csv'.format(dataset))

    # Load mapping
    if 'dbp' in dataset.lower():
        mapping = pd.read_json('./dataset/dbp_type_mapping.json')
    elif 'wd' in dataset.lower() or 'wikidata' in dataset.lower():
        mapping = pd.read_json('./dataset/wd_mapping_type.json')
    else:
        mapping = pd.read_json('./dataset/{}/type_mapping.json'.format(dataset))

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
        model = RandomForestClassifier(verbose=1, n_jobs=-1)
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

    if os.path.isfile('f1_scores/evaluation_rf.csv'):
        df_result.to_csv('./f1_scores/evaluation_rf.csv', mode='a', header=False, index=False)
    else:
        df_result.to_csv('./f1_scores/evaluation_rf.csv', index=False)
