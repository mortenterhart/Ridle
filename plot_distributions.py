import matplotlib.pyplot as plt
import pandas as pd

from dataset_predicate_counts import get_predicate_counts_for_dataset
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def counts_to_list(pred_counts, label):
    predicates = []
    for pred, count in pred_counts[label].iteritems():
        predicates.extend([pred] * count)

    return predicates


def main():
    cached_data = False
    labels = ['wordnet_social_scientist_110619642', 'wordnet_biologist_109855630',
              'wordnet_physicist_110428004', 'wordnet_mathematician_110301261',
              'wordnet_chemist_109913824', 'wordnet_linguist_110264437',
              'wordnet_psychologist_110488865', 'wordnet_geologist_110127689',
              'wordnet_computer_scientist_109951070', 'wordnet_research_worker_110523076']

    # predicate_counts = get_predicate_counts_for_dataset('YAGO-L3-person-scientist')
    print('Computed predicate counts for all classes')
    if cached_data:
        predicate_counts = pd.read_csv('YAGO-L3-person-scientist_predicate_counts.csv')
    else:
        predicate_counts = get_predicate_counts_for_dataset('YAGO-L3-person-scientist')
        predicate_counts.to_csv('YAGO-L3-person-scientist_predicate_counts.csv')

    predicates = list(set([j for i, j in predicate_counts.index.values]))

    encoder = LabelEncoder()
    encoder.fit(predicates)

    predicate_lists = []
    for label in labels:
        print(f'transforming {label}')
        predicate_class_list = counts_to_list(predicate_counts, label)
        predicate_class_list = encoder.transform(predicate_class_list)
        predicate_lists.append(predicate_class_list)
        break

    predicate_distributions = pd.DataFrame({'Class': [labels[0]], 'P': predicate_lists})
    predicate_distributions = predicate_distributions.explode('P')

    print('Created dataframe with predicate distributions')
    print('Dataframe shape:', predicate_distributions.shape)

    sns.displot(data=predicate_distributions, x='P', bins=len(predicates), kind='kde', hue='Class', fill=True)
    print('Created KDE plot')
    plt.show()
    print('plt.show() finished')


if __name__ == '__main__':
    main()
