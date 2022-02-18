import json

import numpy as np
import pandas as pd

from ridle import ROOT_DIR


class SubToClassMapper:
    """
    Maps string subject to string class.
    """

    def __init__(self, mappings):
        """

        Args:
            mappings: Path to the json file with the mappings.
                    Containing idx to class mapping and idx to subject mapping.
        """
        with open(mappings) as json_file:
            data = json.load(json_file)

        self._subject_dict = dict((v, k) for k, v in data['S'].items())
        self._class_dict = data['Class']

    def get_classes(self, subject):
        """
        Get the class name for a subject that is defined in the mappings file/dict.

        Args:
            subject: RDF Subject.

        Returns: List of classes for the subject. (String)

        """
        return self._class_dict[self._subject_dict[subject]]


def get_classes_to_idx_dict(mapper, embeddings):
    """
    Compute a dict to look up the numeric value for a class from the dataset.
    Can be used to convert the classes into a format which is usable for ML training.
    Args:
        embeddings: subjects

    Returns: dict with mappings from string class to numeric representation

    """
    label_set = set()
    for subject in embeddings['S']:
        try:
            classes = mapper.get_classes(subject)
            label_set.update(classes)
        except KeyError:
            print(f'subject {subject} not found in the embeddings')

    class_idx_dict = {class_name: idx for idx, class_name in enumerate(label_set)}

    return class_idx_dict


def class_list_to_numeric(classes, name_to_idx_dict, one_hot=False):
    """
    Convert the list of class names to a list of their numeric representation
    Args:
        classes: list of class names for a subject

    Returns:list of class indices for the subject

    """

    multi_label = [name_to_idx_dict[name] for name in classes]
    if not one_hot:
        return multi_label

    vec_size = len(name_to_idx_dict)
    encoding = np.zeros(vec_size)
    encoding[multi_label] = 1
    return encoding


def main():
    mappings_path = f'{ROOT_DIR}/dataset/dbp_type_mapping.json'
    embeddings_path = f'{ROOT_DIR}/dataset/DBp_2016-04/embedding.csv'
    embeddings = pd.read_csv(embeddings_path)

    # get mapper from subjects to class names (strings)
    mapper = SubToClassMapper(mappings_path)

    # get dict to map class names to a newly defines class index
    class_to_idx_dict = get_classes_to_idx_dict(mapper, embeddings)
    print(class_to_idx_dict)

    # calc multi-label and one-hot encodings
    enc = class_list_to_numeric(['Device', 'Stadium', 'RailwayLine', 'Murderer', 'HistoricBuilding'], class_to_idx_dict)
    print(enc)
    enc = class_list_to_numeric(['Device', 'Stadium', 'RailwayLine', 'Murderer', 'HistoricBuilding'], class_to_idx_dict,
                                one_hot=True)
    print(enc)

    #
    # check which subjects are unmapped for the data set and print the class names for each found subject
    num_unmapped = 0
    label_set = set()

    for subject in embeddings['S']:
        try:
            classes = mapper.get_classes(subject)
            print(subject, ': ', classes)
            label_set.update(classes)
        except KeyError:
            num_unmapped += 1

    print(f'{len(label_set)=}, {label_set}')
    class_idx_dict = {class_name: idx for idx, class_name in enumerate(label_set)}
    print(class_idx_dict)
    print(f'{num_unmapped=}, num_mapped={len(embeddings["S"]) - num_unmapped}')


if __name__ == '__main__':
    main()
