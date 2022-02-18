import argparse

from sklearn.svm import SVC

from ridle.classifier import load_embeddings, evaluate_sklearn_classifier, save_f1_scores


def evaluate_svm(dataset):
    print(f'Training Support Vector Machine on dataset {dataset}:\n')

    embeddings, types = load_embeddings(dataset)

    print('Training SVM...')
    clf = SVC(C=1.0, kernel='rbf', degree=3, verbose=True)
    f1_scores = evaluate_sklearn_classifier(clf, embeddings, types)

    save_f1_scores(f1_scores, dataset, 'svm')


def main():
    parser = argparse.ArgumentParser(description='Support Vector Machine Prediction using Ridle')
    parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
    parser = parser.parse_args()

    dataset = parser.dataset

    evaluate_svm(dataset)


if __name__ == '__main__':
    main()
