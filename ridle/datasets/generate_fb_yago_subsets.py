import os

import pandas as pd

from ridle import ROOT_DIR
from ridle.datasets import fb_yago_subsets
from ridle.evaluation import dataset_class_counts

MIN_CLASS_MEMBERS = 40


def extract_subset(df, include_types):
    class_counts = dataset_class_counts(df[['S', 'Class']].drop_duplicates())

    include_types = [t for t in include_types if t in class_counts.index]
    include_types = [t for t in include_types if class_counts[t] >= MIN_CLASS_MEMBERS]

    return df[df['Class'].isin(include_types)].reset_index(drop=True)


def main():
    fb_triples = pd.read_pickle(f'{ROOT_DIR}/dataset/FB15K237/dataset.pkl')
    fb_types = pd.read_csv(f'{ROOT_DIR}/dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
    fb_labelled = fb_triples.merge(fb_types, on='S', how='inner')

    yago_triples = pd.read_pickle(f'{ROOT_DIR}/dataset/YAGO3-10/dataset.pkl')
    yago_types = pd.read_csv(f'{ROOT_DIR}/dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
    yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], '', regex=True)
    yago_labelled = yago_triples.merge(yago_types, on='S', how='inner')

    for dataset_name, include_types in fb_yago_subsets.items():
        print(f'Extracting dataset {dataset_name}')

        if 'fb' in dataset_name.lower():
            df = fb_labelled
        else:
            df = yago_labelled

        subset = extract_subset(df, include_types)

        if not os.path.exists(f'{ROOT_DIR}/dataset/{dataset_name}'):
            os.makedirs(f'{ROOT_DIR}/dataset/{dataset_name}')

        subset.drop(columns=['Class'], inplace=True)
        subset.to_pickle(f'{ROOT_DIR}/dataset/{dataset_name}/dataset.pkl')
        print(f'Saved dataset {dataset_name}')


if __name__ == '__main__':
    main()
