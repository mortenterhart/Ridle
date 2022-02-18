import pandas as pd

from ridle import ROOT_DIR
from ridle.datasets import fb_yago_subsets
from ridle.utils import aggregate_type_mappings, exclude_external_types


def _preprocess_fb_yago_type_mappings(dataset_name, type_mappings):
    type_mappings = aggregate_type_mappings(type_mappings)

    if dataset_name in fb_yago_subsets:
        type_mappings = exclude_external_types(type_mappings, fb_yago_subsets[dataset_name])

    return type_mappings


def load_type_mappings(dataset_name):
    if 'dbp' in dataset_name.lower():
        mappings = pd.read_json(f'{ROOT_DIR}/dataset/dbp_type_mapping.json')
    elif 'wd' in dataset_name.lower() or 'wikidata' in dataset_name.lower():
        mappings = pd.read_json(f'{ROOT_DIR}/dataset/wd_mapping_type.json')
    elif 'fb' in dataset_name.lower():
        fb_types = pd.read_csv(f'{ROOT_DIR}/dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
        mappings = _preprocess_fb_yago_type_mappings(dataset_name, fb_types)
    elif 'yago' in dataset_name.lower():
        yago_types = pd.read_csv(f'{ROOT_DIR}/dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
        yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], ['', ''], regex=True)
        mappings = _preprocess_fb_yago_type_mappings(dataset_name, yago_types)
    else:
        mappings = pd.read_json(f'{ROOT_DIR}/dataset/{dataset_name}/type_mapping.json')

    return mappings
