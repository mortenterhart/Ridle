import pandas as pd

pd.set_option('display.max_rows', 50)


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
    dataset_names = ['FB-L1', 'FB-L2-org', 'FB-L2-person', 'FB-L3-person-writer', 'YAGO-L1', 'YAGO-L2-org',
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
            mapping = fb_types
        elif 'yago' in dataset.lower():
            yago_types = pd.read_csv('./dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
            yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], ['', ''], regex=True)
            mapping = yago_types
        else:
            mapping = pd.read_json('./dataset/{}/type_mapping.json'.format(dataset))

        # merge them
        print('Processing Data...')
        r = pd.merge(df, mapping, on='S')
        r = r[['S', 'Class']]

        # count the classes
        print(f'Class counts for dataset {dataset}')
        print(dataset_class_counts(r))


if __name__ == '__main__':
    main()
