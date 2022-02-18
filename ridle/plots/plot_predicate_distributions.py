import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from ridle.datasets import fb_yago_subsets
from ridle.evaluation import predicate_counts_for_dataset


def predicate_counts_to_list(predicate_counts, label):
    predicate_list = []
    for pred, count in predicate_counts[label].iteritems():
        predicate_list.extend([pred] * count)

    return predicate_list


def compute_predicate_distributions(dataset_name, include_classes):
    predicate_counts = predicate_counts_for_dataset(dataset_name, include_classes)
    print('Computed predicate counts for all classes')

    predicates = predicate_counts.index.get_level_values('P').unique()

    encoder = LabelEncoder()
    encoder.fit(predicates)

    predicate_lists = []
    for label in include_classes:
        predicate_list = predicate_counts_to_list(predicate_counts, label)
        predicate_list = encoder.transform(predicate_list)
        predicate_lists.append(predicate_list)

    predicate_distributions = pd.DataFrame({'Class': include_classes, 'P': predicate_lists})
    predicate_distributions['Class'].replace(['^wordnet_', '_[0-9]+'], '', regex=True, inplace=True)
    predicate_distributions = predicate_distributions.explode('P').reset_index(drop=True)

    print(f'Created dataframe with predicate distributions of shape {predicate_distributions.shape}')

    return predicate_distributions


def plot_predicate_distributions(predicate_distributions, dataset_name):
    g = sns.displot(data=predicate_distributions, x='P', kind='kde', hue='Class',
                    fill=True, bw_adjust=0.3, aspect=1.5, legend=True)
    g.set(xlabel='Predicate Id', ylabel='Predicate Frequency', title=f'Predicate Distributions for {dataset_name}')
    plt.subplots_adjust(top=0.9)
    plt.show()


def main():
    dataset_name = 'YAGO-L3-person-artist'
    include_classes = fb_yago_subsets[dataset_name]

    predicate_distributions = compute_predicate_distributions(dataset_name, include_classes)
    plot_predicate_distributions(predicate_distributions, dataset_name)


if __name__ == '__main__':
    main()
