import argparse

import pandas as pd


def read_entity_mappings(file_path, open_parameters):
    mappings = {}
    with open(file_path, 'r', **open_parameters) as f:
        # Skip the number of mappings in the first line
        f.readline()

        # Read and store all the mappings
        for line in f:
            entity_name, entity_id = line.strip().split('\t')
            mappings[entity_id] = entity_name

    return mappings


def read_triples(file_path, entity_mappings, relation_mappings, open_parameters):
    triples = []

    with open(file_path, 'r', **open_parameters) as f:
        # Skip the number of triples in the first line
        f.readline()

        # Read and store all the triples
        for line in f:
            s, o, p = line.strip().split()
            subject_name = entity_mappings[s]
            object_name = entity_mappings[o]
            predicate = relation_mappings[p]
            triples.append({'S': subject_name, 'P': predicate, 'O': object_name})

    return pd.DataFrame(triples)


# Load a sample Ridle dataset to showcase the
# required data format for Ridle
ridle_df = pd.read_pickle('./dataset/Company_DBpedia/dataset.pkl')
print('Desired data format for Ridle:')
print(ridle_df.info())
print(ridle_df.head())
print('---------------------------------------------\n')

# Parse the command-line parameters
parser = argparse.ArgumentParser(description='Convert OpenKE datasets to Ridle format')
parser.add_argument('--dataset', type=str, default='FB15K237', nargs='?', help='Dataset to convert')
parser = parser.parse_args()

# Get the dataset to convert
dataset = parser.dataset

# Set special parameters for the open() call and use
# UTF-8 encoding for the YAGO dataset to read correctly
open_parameters = {}
if 'yago' in dataset.lower():
    open_parameters['encoding'] = 'utf-8'

# Read the entity and relation mappings for the OpenKE dataset
entity_mappings = read_entity_mappings(f'./dataset/{dataset}/entity2id.txt', open_parameters)
relation_mappings = read_entity_mappings(f'./dataset/{dataset}/relation2id.txt', open_parameters)
print(f'Read {len(entity_mappings)} entities and {len(relation_mappings)} relations from {dataset} dataset')

# Read the training and test triples from the OpenKE dataset
# using the mappings to resolve the entity names
train_triples = read_triples(f'./dataset/{dataset}/train2id.txt', entity_mappings, relation_mappings, open_parameters)
test_triples = read_triples(f'./dataset/{dataset}/test2id.txt', entity_mappings, relation_mappings, open_parameters)
print(f'Read {len(train_triples)} training and {len(test_triples)} test triples from {dataset} dataset')

# Merge the training and test triples into one dataframe
triples = pd.concat([train_triples, test_triples], ignore_index=True)
print(triples.info())
print(triples.head())

print('---------------------------------------------\n')

# Save the triples to a pickle file
dataset_output_file = f'./dataset/{dataset}/dataset.pkl'
triples.to_pickle(dataset_output_file)

print(f'Saved converted dataset to {dataset_output_file}')
