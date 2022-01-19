import pandas as pd

from type_preprocessing import aggregate_type_mappings, exclude_external_types
from fb_yago_subsets import fb_yago_subsets

pd.set_option('display.max_rows', None)


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
    dataset_names = ['FB-L1', 'FB-L2-org', 'FB-L2-person', 'FB-L3-person-artist', 'YAGO-L1', 'YAGO-L2-org',
                     'YAGO-L2-body_of_water', 'YAGO-L2-person', 'YAGO-L3-person-writer', 'YAGO-L3-person-artist',
                     'YAGO-L3-person-player', 'YAGO-L3-person-scientist']

    for dataset in dataset_names:
        # Load Representations
        print('Reading Data...')
        df = pd.read_csv('./dataset/{}/embedding.csv'.format(dataset))

        # Load mapping
        if 'dbp' in dataset.lower():
            mapping = pd.read_json('./dataset/dbp_type_mapping.json')
        elif 'wd' in dataset.lower() or 'wikidata' in dataset.lower():
            mapping = pd.read_json('./dataset/wd_mapping_type.json')
        elif 'fb' in dataset.lower():
            fb_types = pd.read_csv('./dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
            mapping = aggregate_type_mappings(fb_types)

            if dataset in fb_yago_subsets:
                mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
        elif 'yago' in dataset.lower():
            yago_types = pd.read_csv('./dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
            yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], ['', ''], regex=True)
            mapping = aggregate_type_mappings(yago_types)

            if dataset in fb_yago_subsets:
                mapping = exclude_external_types(mapping, fb_yago_subsets[dataset])
        else:
            mapping = pd.read_json('./dataset/{}/type_mapping.json'.format(dataset))

        # merge them
        r = pd.merge(df, mapping, on='S')
        r = r[['S', 'Class']]

        # print(f'Classes for dataset {dataset}\n{r}')

        # Create one row for each class of entities
        r = r.explode('Class')

        # count the classes
        print(f'Class counts for dataset {dataset} with count < 40')
        class_counts = dataset_class_counts(r)
        print(class_counts[class_counts < 40])


if __name__ == '__main__':
    main()
