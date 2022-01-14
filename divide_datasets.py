import pandas as pd
import os


def extract_dataset(df, labels):
    return pd.concat([df[df['Class'].str.contains('wordnet_' + label + '_[0-9]', regex=True)] for label in labels])


def main():
    fb_triples = pd.read_pickle('dataset/FB15K237/dataset.pkl')
    fb_types = pd.read_csv('dataset/FB15K237/freebase_types.tsv', sep='\t', names=['S', 'Class'])
    fb_labelled = fb_triples.merge(fb_types, on='S', how='inner')

    yago_triples = pd.read_pickle('dataset/YAGO3-10/dataset.pkl')
    yago_types = pd.read_csv('dataset/YAGO3-10/yago_types.tsv', sep='\t', names=['S', 'P', 'Class'])
    yago_types = yago_types[['S', 'Class']].replace(['^<', '>$'], '', regex=True)
    yago_labelled = yago_triples.merge(yago_types, on='S', how='inner')

    # Extract datasets
    fb_level_1 = extract_dataset(fb_labelled, ['person', 'organization', 'body_of_water', 'product'])
    fb_level_2_org = extract_dataset(fb_labelled, ['institution', 'musical_organization', 'party', 'enterprise', 'nongovernmental_organization'])
    fb_level_2_person = extract_dataset(fb_labelled, ['artist', 'politician', 'scientist', 'officeholder', 'writer'])
    fb_level_3_person_writer = extract_dataset(fb_labelled, ['journalist', 'poet', 'novelist', 'scriptwriter', 'dramatist',
                                                             'essayist', 'biographer'])

    yago_level_1 = extract_dataset(yago_labelled, ['person', 'organization', 'body_of_water', 'product'])
    yago_level_2_org = extract_dataset(yago_labelled, ['institution', 'musical_organization', 'party', 'enterprise', 'nongovernmental_organization'])
    yago_level_2_body_of_water = extract_dataset(yago_labelled, ['stream', 'lake', 'ocean', 'bay', 'sea'])
    yago_level_2_person = extract_dataset(yago_labelled, ['artist', 'politician', 'scientist', 'officeholder', 'writer'])
    yago_level_3_person_writer = extract_dataset(yago_labelled, ['journalist', 'poet', 'novelist', 'scriptwriter', 'dramatist',
                                                                 'essayist', 'biographer'])
    yago_level_3_person_artist = extract_dataset(yago_labelled, ['painter', 'sculptor', 'photographer', 'illustrator', 'printmaker'])
    yago_level_3_person_player = extract_dataset(yago_labelled, ['hockey_player', 'soccer_player', 'ballplayer', 'volleyball_player', 'golfer'])
    yago_level_3_person_scientist = extract_dataset(yago_labelled, ['social_scientist', 'biologist', 'physicist',
                                                                    'mathematician', 'chemist', 'linguist', 'psychologist', 'geologist', 'computer_scientist', 'research_worker'])

    # Save datasets in pickle
    dataset_names = ['FB-L1', 'FB-L2-org', 'FB-L2-person', 'FB-L3-person-writer', 'YAGO-L1', 'YAGO-L2-org',
                     'YAGO-L2-body_of_water', 'YAGO-L2-person', 'YAGO-L3-person-writer', 'YAGO-L3-person-artist',
                     'YAGO-L3-person-player', 'YAGO-L3-person-scientist']
    datasets = [fb_level_1, fb_level_2_org, fb_level_2_person, fb_level_3_person_writer, yago_level_1, yago_level_2_org,
                yago_level_2_body_of_water, yago_level_2_person, yago_level_3_person_writer, yago_level_3_person_artist,
                yago_level_3_person_player, yago_level_3_person_scientist]
    for idx, dataset_name in enumerate(dataset_names):
        if not os.path.exists(f'./dataset/{dataset_name}'):
            os.makedirs(f'./dataset/{dataset_name}')

        datasets[idx].drop(columns=['Class'], inplace=True)
        datasets[idx].to_pickle(f'./dataset/{dataset_name}/dataset.pkl')


if __name__ == '__main__':
    main()
