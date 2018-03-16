import numpy as np
import pandas as pd


# FUNCTIONS TO HELP CONSTRUCT EXPERIMENT META DATA ============================
def build_metadata(files, filename_parser):
    """Construct a pandas dataframe of experiment metadata.
    Inputs:
        files - a list of file paths to raw data files
        filename_parser - a function which takes in a filename and returns a named tuple

    Output:
        a pandas dataframe with columns equal to the fields of the named tuple returned
        by the filename_parser function.
    """
    meta = [filename_parser(fname) for fname in files]
    fields = meta[0]._fields
    return pd.DataFrame(meta, columns=fields)


# FUNCTION TO HELP BUILD DATAFRAME OF RAW DATA ================================
def import_raw_data(filenames):
    """Import raw discounting data from a list of filenames.
    Returns a dataframe, each row is one experimental trial"""
    data = []
    for i, fname in enumerate(filenames):
        df = pd.read_csv(fname, sep='\t')
        df = _new_col_of_value(df, 'file_index', i)
        # initials = parse_filename(fname)
        # df = _new_col_of_value(df, 'initials', initials)
        df = _new_col_of_value(df, 'filename', fname)
        data.append(df)

    return(pd.concat(data))


def _new_col_of_value(df, colname, value):
    """Create a new dataframe column with colname, full of value"""
    df[colname] = pd.Series(value, index=df.index)
    return df


def get_data_df_for_a_person(df, initials, cond=None):
    # exact df for just this person and condition
    if cond is None:
        df_one_person = df[[df['initials'] == initials]]
        print('Returning data for both conditions!')
    else:
        df_one_person = df[(df['initials'] == initials)
                           & (df['condition'] == cond)]
    assert len(df_one_person) > 0, "No data!"
    return df_one_person


def get_data_df_for_id_num(df, id):
    # exact df for just this id value
    df_one_person = df[df['id'] == id]
    assert len(df_one_person) > 0, "No data!"
    return df_one_person


def get_data_by_file_index(df, file_index):
    """exact df for just this file_index value"""
    df_one_person = df[df['file_index'] == file_index]
    assert len(df_one_person) > 0, "No data!"
    return df_one_person


def _data_df2dict(df):
    """Convert dataframe to dictionary of np.array
    Note that this is only necessary because A, B, DA, DB make an appearance in
    deterministic nodes.
    It is not a problem to use observed data as pandas dataframes in stochastic
    nodes. You can use them in deterministic nodes, but you have to use call
    the .values, for example data.A.values
    See this for more:
    https://discourse.pymc.io/t/data-frames-vs-numpy-arrays-in-deterministic-nodes/820
    """
    data = {"A": np.array(df.A),
            "DA": np.array(df.DA),
            "B": np.array(df.B),
            "DB": np.array(df.DB),
            "R": np.array(df.R)}
    return data


# MISC DATA RELATED UTILITY FUNCTIONS =========================================
def longest_delay(df):
    """return the longest delay seen in the dataset"""
    return max([max(df.DA), max(df.DB)])
