import pandas as pd

from ridle import ROOT_DIR

if __name__ == '__main__':
    fb = pd.read_pickle(f'{ROOT_DIR}/dataset/FB15K237/dataset.pkl').drop_duplicates(['S'])
    fb_types = pd.read_csv(f'{ROOT_DIR}/dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
    fb_types = fb_types.drop_duplicates(['S'])

    typed_subjects = pd.merge(fb, fb_types, on='S', how='inner')

    print(f'Found types for {len(typed_subjects)} subjects out of {len(fb)} ({len(typed_subjects)/len(fb)*100:.2f}%)')
