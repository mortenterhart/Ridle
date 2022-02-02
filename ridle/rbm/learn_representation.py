import argparse
import os

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
import wget

from ridle import ROOT_DIR
from ridle.rbm import RBM


# https://www.dropbox.com/sh/szvuv79ubfqgmn5/AACHxl_eC0frcGrZpVy0VDQPa?dl=0
download_links = {
    'dblp': 'https://www.dropbox.com/s/78srst5bjt2tta1/dataset.pkl?dl=1',
    'dbp_type_mapping': 'https://www.dropbox.com/s/2ec6dyr90pmjfm9/dbp_type_mapping.json?dl=1',
    'umls': 'https://www.dropbox.com/s/madbrirjc3yjtru/dataset.pkl?dl=1',
    'Person_DBpedia': 'https://www.dropbox.com/s/1omj2btnoj8g4xa/dataset.pkl?dl=1',
    'DBp_2016-04': 'https://www.dropbox.com/s/z38exis1ah3q5ze/dataset.pkl?dl=1',
    'Company_DBpedia': 'https://www.dropbox.com/s/bft3hmk2m6ecrkl/dataset.pkl?dl=1',
    'Songs_DBpedia': 'https://www.dropbox.com/s/u9k6qaydqowckae/dataset.pkl?dl=1',
    'Books_DBpedia': 'https://www.dropbox.com/s/wdqhov2g4bvwzr9/dataset.pkl?dl=1',
    'ChemicalCompounds_DBpedia': 'https://www.dropbox.com/s/fyyqgtwwf2pnj3b/dataset.pkl?dl=1',
    'Universities_DBpedia': 'https://www.dropbox.com/s/0g2moh3puz09uoy/dataset.pkl?dl=1'
}


def download_data(dataset):
    if dataset not in download_links:
        raise ValueError(f'Dataset {dataset} not available')

    if not os.path.isfile(f'{ROOT_DIR}/dataset/dbp_type_mapping.json'):
        print("Downloading dbp_type_mapping data.")
        data_url = download_links['dbp_type_mapping']
        wget.download(data_url, f'{ROOT_DIR}/dataset/dbp_type_mapping.json')

    if not os.path.isfile(f'{ROOT_DIR}/dataset/{dataset}/dataset.pkl'):
        print(f'Downloading {dataset} data.')
        data_url = download_links[dataset]
        Path(f'{ROOT_DIR}/dataset/{dataset}').mkdir(parents=True, exist_ok=True)
        wget.download(data_url, f'{ROOT_DIR}/dataset/{dataset}/dataset.pkl')


def learn_representation(dataset):
    if not os.path.isfile(f'{ROOT_DIR}/dataset/{dataset}/dataset.pkl'):
        raise FileNotFoundError(f'Dataset {dataset} not found')

    print(f'Learning Ridle Representations on {dataset}')

    # Load dataset
    df = pd.read_pickle(f'{ROOT_DIR}/dataset/{dataset}/dataset.pkl')[['S', 'P']].drop_duplicates()

    # Learn representation
    mlb = MultiLabelBinarizer()
    mlb.fit([df['P'].unique()])
    df_distr_s = df.groupby('S')['P'].apply(list).reset_index(name='Class')
    X = mlb.transform(df_distr_s['Class'])
    rbm = RBM(n_hidden=50, n_iterations=100, batch_size=100, learning_rate=0.01)
    rbm.fit(X)

    # Save Entity Representation
    r = pd.DataFrame(rbm.compress(X), index=df_distr_s['S']).reset_index()
    r.to_csv(f'{ROOT_DIR}/dataset/{dataset}/embedding.csv', index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Ridle, learning a representation for entities using a '
                    'target distributions over the usage of relations.',
    )
    parser.add_argument('--dataset', nargs='?', default='DBp_2016-04', type=str)
    parser = parser.parse_args()

    dataset = parser.dataset

    download_data(dataset)
    learn_representation(dataset)


if __name__ == '__main__':
    main()
