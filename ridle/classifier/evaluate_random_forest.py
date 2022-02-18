import argparse

from sklearn.ensemble import RandomForestClassifier

from ridle.classifier import load_embeddings, evaluate_sklearn_classifier, save_f1_scores


def evaluate_random_forest(dataset):
    print(f'Training Random Forest on dataset {dataset}:\n')

    embeddings, types = load_embeddings(dataset)

    print('Training Random Forest...')
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, verbose=1, n_jobs=-1)
    f1_scores = evaluate_sklearn_classifier(clf, embeddings, types)

    save_f1_scores(f1_scores, dataset, 'rf')


def main():
    parser = argparse.ArgumentParser(description='Random Forest Prediction using Ridle')
    parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
    parser = parser.parse_args()

    dataset = parser.dataset

    evaluate_random_forest(dataset)


if __name__ == '__main__':
    main()
