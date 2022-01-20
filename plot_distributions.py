import matplotlib.pyplot as plt
from dataset_predicate_counts import get_predicate_counts_for_dataset


def main():
    labels = ['wordnet_social_scientist_110619642', 'wordnet_biologist_109855630',
              'wordnet_physicist_110428004', 'wordnet_mathematician_110301261',
              'wordnet_chemist_109913824', 'wordnet_linguist_110264437',
              'wordnet_psychologist_110488865', 'wordnet_geologist_110127689',
              'wordnet_computer_scientist_109951070', 'wordnet_research_worker_110523076']

    predicate_counts = get_predicate_counts_for_dataset('YAGO-L3-person-scientist')

    for label in labels:
        class_predicate_counts = predicate_counts[label].values
        print(class_predicate_counts)

        plt.plot(class_predicate_counts)
        plt.title(label)
        plt.show()


if __name__ == '__main__':
    main()
