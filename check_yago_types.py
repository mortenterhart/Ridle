import pandas as pd

if __name__ == '__main__':
    yago = pd.read_pickle('dataset/YAGO3-10/dataset.pkl').drop_duplicates(['S'])
    yago_types = pd.read_csv('dataset/yago/SimpleTypeFactsWordnetLevel.tsv', sep='\t', names=['S', 'P', 'Class'])
    yago_types = yago_types[['S', 'Class']].drop_duplicates(['S'])
    yago_types = yago_types.replace(['^<', '>$'], ['', ''], regex=True)

    typed_subjects = pd.merge(yago, yago_types, on='S', how='inner')

    print(f'Found types for {len(typed_subjects)} subjects out of {len(yago)} '
          f'({len(typed_subjects) / len(yago) * 100:.2f}%)')
