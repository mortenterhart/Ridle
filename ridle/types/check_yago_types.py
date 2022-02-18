import pandas as pd

from ridle import ROOT_DIR
from ridle.types import load_type_mappings


def main():
    yago = pd.read_pickle(f'{ROOT_DIR}/dataset/YAGO3-10/dataset.pkl').drop_duplicates(['S'])
    yago_types = load_type_mappings('YAGO3-10')

    typed_subjects = pd.merge(yago, yago_types, on='S', how='inner')

    print(f'Found types for {len(typed_subjects)} subjects out of {len(yago)} '
          f'({len(typed_subjects) / len(yago) * 100:.2f}%)')


if __name__ == '__main__':
    main()
