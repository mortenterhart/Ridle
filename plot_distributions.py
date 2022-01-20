import os

import matplotlib.pyplot as plt
import pandas as pd

from fb_yago_subsets import fb_yago_subsets
from dataset_predicate_counts import get_predicate_counts_for_dataset
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import re


def counts_to_list(pred_counts, label):
    predicates = []
    for pred, count in pred_counts[label].iteritems():
        predicates.extend([pred] * count)

    return predicates


def main():
    dataset_name = 'YAGO-L3-person-scientist'

    labels = fb_yago_subsets[dataset_name]

    if os.path.exists(f'{dataset_name}_predicate_counts.pkl'):
        predicate_counts = pd.read_pickle(f'{dataset_name}_predicate_counts.pkl')
    else:
        predicate_counts = get_predicate_counts_for_dataset(dataset_name)
        predicate_counts.to_pickle(f'{dataset_name}_predicate_counts.pkl')

    print('Computed predicate counts for all classes')

    predicates = list(set([j for i, j in predicate_counts.index.values]))

    encoder = LabelEncoder()
    encoder.fit(predicates)

    predicate_lists = []
    for label in labels:
        print(f'transforming {label}')
        predicate_class_list = counts_to_list(predicate_counts, label)
        predicate_class_list = encoder.transform(predicate_class_list)
        predicate_lists.append(predicate_class_list)

    predicate_distributions = pd.DataFrame({'Class': labels, 'P': predicate_lists})
    predicate_distributions = predicate_distributions.explode('P').reset_index(drop=True)
    predicate_distributions['Class'].replace(['^wordnet_', '_[0-9]+'], '', regex=True, inplace=True)

    print('Created dataframe with predicate distributions')
    print('Dataframe shape:', predicate_distributions.shape)

    g = sns.displot(data=predicate_distributions, x='P', kind='kde', hue='Class',
                    fill=True, bw_adjust=0.3, aspect=1.5, legend=True)
    # plt.legend(loc='upper left', labels=[re.sub("^wordnet_|_[0-9]+", "", i) for i in reversed(labels)], title='Class',
    #            frameon=False)
    g.set(xlabel="Predicate Id", ylabel="Predicate Frequency", title=f'Predicate Distributions for {dataset_name}')
    plt.subplots_adjust(top=0.9)
    plt.show()


if __name__ == '__main__':
    main()
