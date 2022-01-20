import pandas as pd

from type_preprocessing import aggregate_type_mappings, exclude_external_types
from fb_yago_subsets import fb_yago_subsets

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_predicate_counts_per_class(triples_df, include_types):
    reduced_triples = triples_df[['S', 'P', 'Class']].drop_duplicates()

    # Filter out triples that are not in the include_types
    reduced_triples = reduced_triples[reduced_triples['Class'].isin(include_types)]

    return reduced_triples.groupby(['Class', 'P']).size()


def get_predicate_counts_for_dataset(dataset_name):
    triples = pd.read_pickle(f'dataset/{dataset_name}/dataset.pkl')

    # Load mapping
    if 'dbp' in dataset_name.lower():
        mapping = pd.read_json('./dataset/dbp_type_mapping.json')
    elif 'wd' in dataset_name.lower() or 'wikidata' in dataset_name.lower():
        mapping = pd.read_json('./dataset/wd_mapping_type.json')
    elif 'fb' in dataset_name.lower():
        fb_types = pd.read_csv('./dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
        mapping = aggregate_type_mappings(fb_types)

        if dataset_name in fb_yago_subsets:
            mapping = exclude_external_types(mapping, fb_yago_subsets[dataset_name])
    elif 'yago' in dataset_name.lower():
        yago_types = pd.read_csv('./dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
        yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], ['', ''], regex=True)
        mapping = aggregate_type_mappings(yago_types)

        if dataset_name in fb_yago_subsets:
            mapping = exclude_external_types(mapping, fb_yago_subsets[dataset_name])
    else:
        mapping = pd.read_json('./dataset/{}/type_mapping.json'.format(dataset_name))

    # Unfold list of types into multiple rows
    mapping = mapping.explode('Class')

    # Join the triples with the type mappings
    labelled_triples = triples.merge(mapping, on='S', how='inner')

    return get_predicate_counts_per_class(labelled_triples, fb_yago_subsets[dataset_name])


def main():
    for dataset in fb_yago_subsets.keys():
        predicate_counts = get_predicate_counts_for_dataset(dataset)
        print(predicate_counts)


if __name__ == '__main__':
    main()
