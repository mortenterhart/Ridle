import pandas as pd

from ridle import ROOT_DIR
from ridle.types import load_type_mappings


def dataset_class_counts(triples_df):
    """
    Returns the number of instances of each class in the dataset.

    Parameters
    ----------
    triples_df : pandas.DataFrame
        The dataset to be analyzed.

    Returns
    -------
    pandas.Series
        A series with the number of instances of each class.
    """
    return triples_df.groupby('Class').size().sort_values()


def main():
    dataset_names = ['Songs_DBpedia', 'Person_DBpedia', 'Universities_DBpedia', 'ChemicalCompounds_DBpedia',
                     'Books_DBpedia', 'umls', 'dblp', 'YAGO3-10']

    for dataset in dataset_names:
        # Load Representations
        print(f'Loading dataset {dataset}...')
        df = pd.read_csv(f'{ROOT_DIR}/dataset/{dataset}/embedding.csv')

        # Load type mappings
        mappings = load_type_mappings(dataset)

        # Merge them
        mapped = pd.merge(df, mappings, on='S')
        mapped = mapped[['S', 'Class']]

        # Create one row for each class of entities
        mapped = mapped.explode('Class')

        # Count the classes
        print(f'Class counts for dataset {dataset}')
        class_counts = dataset_class_counts(mapped)
        print(class_counts)


if __name__ == '__main__':
    main()
