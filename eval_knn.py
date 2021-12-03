import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer


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
    description='Instance Type Prediction using Ridle',
)
parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
parser = parser.parse_args()

# Load Representations
print('Reading Data...')
df = pd.read_csv('./dataset/{}/embedding.csv'.format(parser.dataset))

# Load mapping
if 'dbp' in parser.dataset.lower():
    mapping = pd.read_json('./dataset/dbp_type_mapping.json')
elif 'wd' in parser.dataset.lower() or 'wikidata' in parser.dataset.lower():
    mapping = pd.read_json('./dataset/wd_mapping_type.json')
else:
    mapping = pd.read_json('./dataset/{}/type_mapping.json'.format(parser.dataset))

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