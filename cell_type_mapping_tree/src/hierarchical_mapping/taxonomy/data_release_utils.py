"""
This module will contain utility functions to help read the CSV files that
are part of an official data release taxonomy.
"""


def get_header_map(
        csv_path,
        desired_columns):
    """
    Return a dict mapping the name of columns to the
    index of the columns indicating where they appear in
    a csv

    Parameters
    ----------
    csv_path:
        Path to the csv
    desired_columns:
        List of the column_names whose locations you want
    """
    error_msg = ''
    with open(csv_path, 'r') as src:
        header_line = src.readline()
    header_line = header_line.strip().split(',')
    desired_columns = set(desired_columns)
    result = dict()
    for idx, value in enumerate(header_line):
        if value in desired_columns:
            if value in result:
                error_msg += f"column '{value}' occurs more than once\n"
            result[value] = idx
    for expected in desired_columns:
        if expected not in result:
            error_msg += f"could not find column '{expected}'\n"
    if len(error_msg) > 0:
        error_msg = f"errors parsing {csv_path}:\n{error_msg}"
        raise RuntimeError(error_msg)
    return result
