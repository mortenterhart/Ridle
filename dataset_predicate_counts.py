import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_predicate_counts(triples_df):
    return triples_df['P'].value_counts()


def main():
    dataset_names = ['FB-L1', 'FB-L2-org', 'FB-L2-person', 'FB-L3-person-writer', 'YAGO-L1', 'YAGO-L2-org',
                     'YAGO-L2-body_of_water', 'YAGO-L2-person', 'YAGO-L3-person-writer', 'YAGO-L3-person-artist',
                     'YAGO-L3-person-player', 'YAGO-L3-person-scientist']

    for dataset in dataset_names:
        triples = pd.read_pickle(f'dataset/{dataset}/dataset.pkl')
        print(f'{len(triples)} triples in {dataset}')
        print('Predicate counts:')
        print(get_predicate_counts(triples))


if __name__ == '__main__':
    main()
