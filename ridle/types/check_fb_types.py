import pandas as pd

from ridle import ROOT_DIR
from ridle.types import load_type_mappings


def main():
    fb = pd.read_pickle(f'{ROOT_DIR}/dataset/FB15K237/dataset.pkl').drop_duplicates(['S'])
    fb_types = load_type_mappings('FB15K237')

    typed_subjects = pd.merge(fb, fb_types, on='S', how='inner')

    print(f'Found types for {len(typed_subjects)} subjects out of {len(fb)} '
          f'({len(typed_subjects) / len(fb) * 100:.2f}%)')


if __name__ == '__main__':
    main()
