import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import clone_model

from ridle import ROOT_DIR
from ridle.types import load_type_mappings

K_FOLD = 10


def load_embeddings(dataset):
    # Load embeddings
    print(f'Loading dataset {dataset}...')
    df = pd.read_csv(f'{ROOT_DIR}/dataset/{dataset}/embedding.csv')

    # Load type mappings
    mappings = load_type_mappings(dataset)

    # Merge embeddings and type mappings
    print('Processing data...')
    mapped = pd.merge(df, mappings, on='S')

    embeddings = mapped.drop(['S', 'Class'], axis=1).values
    types = mapped['Class'].values

    return embeddings, types


def evaluate_sklearn_classifier(clf, embeddings, labels):
    mlb = MultiLabelBinarizer()
    targets = mlb.fit_transform(labels)

    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
    fold_no = 1

    accuracies = []
    f1_scores = {
        'macro': [],
        'micro': [],
        'weighted': []
    }

    for train, test in kfold.split(embeddings, targets):
        clf = clone(clf)

        clf.fit(embeddings[train], targets[train])
        y_pred = clf.predict(embeddings[test])

        accuracies.append(accuracy_score(targets[test], y_pred))
        f1_scores['macro'].append(f1_score(targets[test], y_pred, average='macro', zero_division=1))
        f1_scores['micro'].append(f1_score(targets[test], y_pred, average='micro', zero_division=1))
        f1_scores['weighted'].append(f1_score(targets[test], y_pred, average='weighted', zero_division=1))

        _print_fold_scores(fold_no, f1_scores, accuracies)
        fold_no += 1

    _print_average_scores(f1_scores, accuracies)

    return f1_scores


def evaluate_keras_classifier(model, embeddings, labels):
    mlb = MultiLabelBinarizer()
    targets = mlb.fit_transform(labels)

    kfold = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
    fold_no = 1

    accuracies = []
    losses = []
    f1_scores = {
        'macro': [],
        'micro': [],
        'weighted': []
    }

    for train, test in kfold.split(embeddings, targets):
        model = clone_model(model)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        model.fit(embeddings[train], targets[train],
                  batch_size=64,
                  validation_data=(embeddings[test], targets[test]),
                  epochs=100)
        y_pred = model.predict(embeddings[test])
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        metrics = model.evaluate(embeddings[test], targets[test])
        losses.append(metrics[0])
        accuracies.append(metrics[1])
        f1_scores['macro'].append(f1_score(targets[test], y_pred, average='macro', zero_division=1))
        f1_scores['micro'].append(f1_score(targets[test], y_pred, average='micro', zero_division=1))
        f1_scores['weighted'].append(f1_score(targets[test], y_pred, average='weighted', zero_division=1))

        _print_fold_scores(fold_no, f1_scores, accuracies, losses)

        fold_no += 1

    _print_average_scores(f1_scores, accuracies, losses)

    return f1_scores


def save_f1_scores(f1_scores, dataset, clf_name):
    result_df = pd.DataFrame({
        'F1-Macro': np.mean(f1_scores['macro']),
        'F1-Macro_std': np.std(f1_scores['macro']),
        'F1-Micro': np.mean(f1_scores['micro']),
        'F1-Micro_std': np.std(f1_scores['micro']),
        'F1-Weighted': np.mean(f1_scores['weighted']),
        'F1-Weighted_std': np.std(f1_scores['weighted']),
        'Dataset': dataset,
        'Classifier': clf_name,
        'Method': 'Ridle'
    })
    print(result_df)

    if os.path.isfile(f'{ROOT_DIR}/f1_scores/evaluation_{clf_name}.csv'):
        result_df.to_csv(f'{ROOT_DIR}/f1_scores/evaluation_{clf_name}.csv', mode='a', header=False, index=False)
    else:
        result_df.to_csv(f'{ROOT_DIR}/f1_scores/evaluation_{clf_name}.csv', index=False)


def _print_fold_scores(fold_no, f1_scores, accuracies, losses=None):
    print(f'\nScores for fold {fold_no}:')
    print(f'Loss:        {losses[-1]}') if losses is not None else None
    print(f'Accuracy:    {accuracies[-1]}')
    print(f'F1-Macro:    {f1_scores["macro"][-1]}')
    print(f'F1-Micro:    {f1_scores["micro"][-1]}')
    print(f'F1-Weighted: {f1_scores["weighted"][-1]}\n')


def _print_average_scores(f1_scores, accuracies, losses=None):
    print('------------------------------------------------------------------------')
    print(f'Average scores over all folds:')
    print(f'Loss:        {np.mean(losses)} (+/- {np.std(losses)})') if losses is not None else None
    print(f'Accuracy:    {np.mean(accuracies)} (+/- {np.std(accuracies)})')
    print(f'F1-Macro:    {np.mean(f1_scores["macro"])} (+/- {np.std(f1_scores["macro"])})')
    print(f'F1-Micro:    {np.mean(f1_scores["micro"])} (+/- {np.std(f1_scores["micro"])})')
    print(f'F1-Weighted: {np.mean(f1_scores["weighted"])} (+/- {np.std(f1_scores["weighted"])})')
    print('------------------------------------------------------------------------')
