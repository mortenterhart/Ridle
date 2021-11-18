import json
import pandas as pd


class SubToClassMapper:
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
        Get the classes for a subject that is defined in the mappings file/dict.

        Args:
            subject: RDF Subject.

        Returns: List of classes for the subject.

        """
        return self._class_dict[self._subject_dict[subject]]


if __name__ == '__main__':
    mappings_path = "dataset/dbp_type_mapping.json"
    embeddings_path = "dataset/DBp_2016-04/embedding.csv"

    mapper = SubToClassMapper(mappings_path)

    embeddings = pd.read_csv(embeddings_path)
    num_unmapped = 0
    for subject in embeddings['S']:
        try:
            print(subject, ": ", mapper.get_classes(subject))
        except KeyError:
            num_unmapped += 1

    print(f"{num_unmapped=}, num_mapped={len(embeddings['S']) - num_unmapped}")
