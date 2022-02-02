import pandas as pd

from ridle import ROOT_DIR
from ridle.utils import aggregate_type_mappings, exclude_external_types
from ridle.datasets import fb_yago_subsets

pd.set_option('display.max_rows', 100)


def dataset_class_counts(types_df):
    """
    Returns the number of instances of each class in the dataset.

    Parameters
    ----------
    types_df : pandas.DataFrame
        The dataset to be analyzed.

    Returns
    -------
    pandas.Series
        A series with the number of instances of each class.
    """
    return types_df.groupby('Class').size().sort_values()


def main():
    dataset_names = ['Songs_DBpedia', 'Person_DBpedia', 'Universities_DBpedia', 'ChemicalCompounds_DBpedia',
                     'Books_DBpedia', 'umls', 'dblp', 'YAGO3-10']

    for dataset in dataset_names:
        # Load Representations
        print('Reading Data...')
        df = pd.read_csv(f'{ROOT_DIR}/dataset/{dataset}/embedding.csv')

        # Load mapping
        if 'dbp' in dataset.lower():
            mapping = pd.read_json(f'{ROOT_DIR}/dataset/dbp_type_mapping.json')
        elif 'wd' in dataset.lower() or 'wikidata' in dataset.lower():
            mapping = pd.read_json(f'{ROOT_DIR}/dataset/wd_mapping_type.json')
        elif 'fb' in dataset.lower():
            fb_types = pd.read_csv(f'{ROOT_DIR}/dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
            mapping = aggregate_type_mappings(fb_types)

            if dataset in fb_yago_subsets:
                mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
        elif 'yago' in dataset.lower():
            yago_types = pd.read_csv(f'{ROOT_DIR}/dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
            yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], ['', ''], regex=True)
            mapping = aggregate_type_mappings(yago_types)

            if dataset in fb_yago_subsets:
                mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
        else:
            mapping = pd.read_json(f'{ROOT_DIR}/dataset/{dataset}/type_mapping.json')

        # merge them
        r = pd.merge(df, mapping, on='S')
        r = r[['S', 'Class']]

        # Create one row for each class of entities
        r = r.explode('Class')

        # count the classes
        print(f'Class counts for dataset {dataset}')
        class_counts = dataset_class_counts(r)
        print(class_counts)


if __name__ == '__main__':
    main()
