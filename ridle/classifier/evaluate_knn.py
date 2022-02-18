import argparse

from sklearn.neighbors import KNeighborsClassifier

from ridle.classifier import load_embeddings, evaluate_sklearn_classifier, save_f1_scores


def evaluate_knn(dataset):
    print(f'Training KNN on dataset {dataset}:\n')

    embeddings, types = load_embeddings(dataset)

    print('Training KNN...')
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    f1_scores = evaluate_sklearn_classifier(clf, embeddings, types)

    save_f1_scores(f1_scores, dataset, 'knn')


def main():
    parser = argparse.ArgumentParser(description='K-nearest Neighbors Prediction using Ridle')
    parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
    parser = parser.parse_args()

    dataset = parser.dataset

    evaluate_knn(dataset)


if __name__ == '__main__':
    main()
