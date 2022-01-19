import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from type_preprocessing import aggregate_type_mappings, exclude_external_types
from fb_yago_subsets import fb_yago_subsets


class MultiLabelKNN:
    def __init__(self, n):
        self.n = n
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        if X is None:
            raise ValueError('No points to predict')

        if self.X is None:
            raise ValueError('No training data given')

        if self.y is None:
            raise ValueError('No labels given')

        # Get euclidean distances of all points to training points
        dist_matrix = euclidean_distances(X, self.X)

        neighbors = np.argpartition(dist_matrix, self.n, axis=0)[:, :self.n]
        label_counts = np.empty((X.shape[0], self.y.shape[1]))

        for i, neighbor in enumerate(neighbors):
            label = self.y[neighbor].sum(axis=0)
            label_counts[i] = label

        return label_counts


parser = argparse.ArgumentParser(
    description='K-nearest Neighbors Prediction using Ridle',
)
parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
parser = parser.parse_args()

dataset = parser.dataset

dataset_names = ['FB-L1', 'FB-L2-org', 'FB-L2-person', 'FB-L3-person-artist', 'YAGO-L1', 'YAGO-L2-org',
                 'YAGO-L2-body_of_water', 'YAGO-L2-person', 'YAGO-L3-person-writer', 'YAGO-L3-person-artist',
                 'YAGO-L3-person-player', 'YAGO-L3-person-scientist']

for dataset in dataset_names:
    print(f'Training on dataset {dataset}')

    # Load Representations
    print('Reading Data...')
    df = pd.read_csv('./dataset/{}/embedding.csv'.format(dataset))

    # Load mapping
    if 'dbp' in dataset.lower():
        mapping = pd.read_json('./dataset/dbp_type_mapping.json')
    elif 'wd' in dataset.lower() or 'wikidata' in dataset.lower():
        mapping = pd.read_json('./dataset/wd_mapping_type.json')
    elif 'fb' in dataset.lower():
        fb_types = pd.read_csv('./dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
        mapping = aggregate_type_mappings(fb_types)

        if dataset in fb_yago_subsets:
            mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
    elif 'yago' in dataset.lower():
        yago_types = pd.read_csv('./dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
        yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], ['', ''], regex=True)
        mapping = aggregate_type_mappings(yago_types)

        if dataset in fb_yago_subsets:
            mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
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

        clf = KNeighborsClassifier(n_jobs=-1)
        clf.fit(inputs[train], targets[train])
        print(clf.score(inputs[test], targets[test]))
        y_pred = clf.predict(inputs[test])
        print(y_pred)

        # n = 6
        # knn = MultiLabelKNN(n=n)
        # knn.fit(inputs[train], targets[train])
        # y_pred = knn.predict(inputs[test])
        #
        # print(y_pred[:3])
        #
        # fold_no += 1
        #
        # # y_pred = softmax(y_pred, axis=1)
        # # targets_softmax = softmax(targets[test], axis=1)
        #
        # # errors = np.sum(np.abs(y_pred - targets_softmax), axis=1)
        #
        # pred_thresholds = np.empty((y_pred.shape[0]))
        # for i, pred in enumerate(y_pred):
        #     pred_thresholds[i] = np.mean(pred[pred > 0])
        #
        # # pred_thresholds = np.mean(y_pred[y_pred > 0], axis=1)
        # print(f'thresholds {pred_thresholds.shape}: {pred_thresholds[:3]}')
        # for i, pred_threshold in enumerate(pred_thresholds):
        #     y_pred[i][y_pred[i] / float(n) >= 0.5] = 1
        #     y_pred[i][y_pred[i] / float(n) < 0.5] = 0
        #
        # # print(f"errors: {errors[:3]}")
        accuracy = accuracy_score(targets[test], y_pred)

        f1_macro.append(f1_score(targets[test], y_pred, average='macro', zero_division=1))
        f1_micro.append(f1_score(targets[test], y_pred, average='micro', zero_division=1))
        f1_weighted.append(f1_score(targets[test], y_pred, average='weighted', zero_division=1))

        print(f"accuracy: {accuracy}")
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

    if os.path.isfile('f1_scores/evaluation_knn.csv'):
        df_result.to_csv('./f1_scores/evaluation_knn.csv', mode='a', header=False, index=False)
    else:
        df_result.to_csv('./f1_scores/evaluation_knn.csv', index=False)
