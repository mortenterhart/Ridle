import pandas as pd
import os

from fb_yago_subsets import fb_yago_subsets
from dataset_class_counts import dataset_class_counts

MIN_CLASS_MEMBERS = 40


def extract_dataset(df, labels):
    class_counts = dataset_class_counts(df[['S', 'Class']].drop_duplicates())
    labels = set(class_counts.index) & set(labels)
    class_counts = class_counts[list(labels)]
    include_types = class_counts[class_counts >= MIN_CLASS_MEMBERS].index
    return pd.concat([df[df['Class'] == label] for label in include_types]).reset_index(drop=True)


def main():
    fb_triples = pd.read_pickle('dataset/FB15K237/dataset.pkl')
    fb_types = pd.read_csv('dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
    fb_labelled = fb_triples.merge(fb_types, on='S', how='inner')

    yago_triples = pd.read_pickle('dataset/YAGO3-10/dataset.pkl')
    yago_types = pd.read_csv('dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
    yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], '', regex=True)
    yago_labelled = yago_triples.merge(yago_types, on='S', how='inner')

    for dataset_name, include_types in fb_yago_subsets.items():
        print(f'Extracting dataset {dataset_name}')

        dataset = fb_labelled if 'fb' in dataset_name.lower() else yago_labelled
        subset = extract_dataset(dataset, include_types)

        if not os.path.exists(f'./dataset/{dataset_name}'):
            os.makedirs(f'./dataset/{dataset_name}')

        subset.drop(columns=['Class'], inplace=True)
        subset.to_pickle(f'./dataset/{dataset_name}/dataset.pkl')
        print(f'Saved dataset {dataset_name}')


if __name__ == '__main__':
    main()
