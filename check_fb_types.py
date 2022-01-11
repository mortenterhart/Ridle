import pandas as pd

if __name__ == '__main__':
    fb = pd.read_pickle('dataset/YAGO3-10/dataset.pkl')
    fb_types = pd.read_csv('dataset/yago/SimpleTypeFactsWordnetLevel.tsv', sep='\t', names=['S', 'P', 'Class'])
    fb_types = fb_types.drop_duplicates(['S'])
    fb_types = fb_types.replace(['^<', '>$'], ['', ''], regex=True)

    typed_subjects = pd.merge(fb, fb_types, on='S', how='inner')

    print(f'Found types for {len(typed_subjects)} subjects out of {len(fb)} ({len(typed_subjects)/len(fb)*100:.2f}%)')
