import pandas as pd


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
    types_df['Class'] = types_df['Class'].apply(lambda types: [t for t in types if t in include_types])
    return types_df

