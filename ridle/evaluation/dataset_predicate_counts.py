import pandas as pd

from ridle import ROOT_DIR
from ridle.datasets import fb_yago_subsets
from ridle.types import load_type_mappings


def predicate_counts_per_class(triples_df, include_classes):
    reduced_triples = triples_df[['S', 'P', 'Class']].drop_duplicates()

    # Filter out triples that are not in include_classes
    reduced_triples = reduced_triples[reduced_triples['Class'].isin(include_classes)]

    return reduced_triples.groupby(['Class', 'P']).size()


def predicate_counts_for_dataset(dataset_name, include_classes):
    # Load the dataset
    triples = pd.read_pickle(f'{ROOT_DIR}/dataset/{dataset_name}/dataset.pkl')

    # Load the type mappings
    mappings = load_type_mappings(dataset_name)

    # Unfold list of types into multiple rows
    mappings = mappings.explode('Class')

    # Join the triples with the type mappings
    labelled_triples = triples.merge(mappings, on='S', how='inner')

    return predicate_counts_per_class(labelled_triples, include_classes)


def main():
    pd.set_option('display.max_rows', 100)

    for dataset, include_classes in fb_yago_subsets.items():
        predicate_counts = predicate_counts_for_dataset(dataset, include_classes)
        print(predicate_counts)


if __name__ == '__main__':
    main()
