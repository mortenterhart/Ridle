import pandas as pd
import re


def aggregate_type_mappings(types_df):
    """
    Aggregates the types of each entity in a dataframe
    from multiple rows into a list of types in a single
    row. The given types_df dataframe has one type per row.
    Args:
        types_df: dataframe with entities and one type per row

    Returns: a dataframe with aggregated types for each entity
    """
    type_mappings = {}

    for _, mapping in types_df.iterrows():
        if mapping['S'] in type_mappings:
            type_mappings[mapping['S']].append(mapping['Class'])
        else:
            type_mappings[mapping['S']] = [mapping['Class']]

    return pd.DataFrame({'S': type_mappings.keys(), 'Class': type_mappings.values()})


def exclude_external_types(types_df, include_types):
    include_regexes = [re.compile('wordnet_' + include_type + '_[0-9]+') for include_type in include_types]

    def contained_elems(types):
        res = []

        for t in types:
            for regex in include_regexes:
                if regex.search(t):
                    res.append(t)
                    break

        return res

    types_df['Class'] = types_df['Class'].apply(lambda types: contained_elems(types))
    return types_df

