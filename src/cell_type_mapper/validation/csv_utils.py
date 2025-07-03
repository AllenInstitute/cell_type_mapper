"""
Define utility functions for accepting CSV inputs to MapMyCells
"""
import anndata
import gzip
import numpy as np
import pandas as pd
import pathlib
import traceback
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    get_timestamp
)


def convert_csv_to_h5ad(
        src_path,
        log):
    """
    Convert CSV file to h5ad file (if necessary)

    Parameters
    ----------
    src_path:
        Path to the src file
    log:
        Optional logger to log messages for CLI

    Returns
    -------
    h5ad_path:
        Path to the h5ad file (this will be src_path
        if src_path does not end in '.csv' or '.gz')
    was_converted:
        Boolean indicating if the file was converted to an
        h5ad file

    Notes
    -----
    This function has to determine whether or not to set first_column_names
    to True when reading the CSV with anndata (i.e. it has to determine if
    the first column in the file is a list of cell labels, or is just another
    gene).

    To make this determination, it applies the following test:

    - If the first entry in the header column is '', then
       first_column_names=True

    - If the first entry in the first data row cannot be converted to a float,
      (i.e. if it is just a string), then first_column_names=True

    - Otherwise, we assume the file is purely made up of gene expression data
      and first_column_names=False

    We believe this will be save because, even if th first column is supposed
    to be cell labels and those labels are numerical, the first column header
    (which must, in this case, not be blank) should not map to any gene
    identifiers.
    """
    src_path = pathlib.Path(src_path)
    src_name = src_path.name

    if not src_name.endswith('.csv') and not src_name.endswith('.gz'):
        return (src_path, False)

    if src_name.endswith('.gz'):
        src_suffix = '.gz'
    elif src_name.endswith('.csv'):
        src_suffix = '.csv'

    dst_name = src_name.replace(src_suffix, '')
    dst_name = f'{dst_name}-{get_timestamp()}.h5ad'
    dst_path = src_path.parent/dst_name

    if dst_path.exists():
        dst_path = mkstemp_clean(
            dir=src_path.parent,
            prefix=src_name.replace(src_suffix, '_'),
            suffix='.h5ad'
        )

    warning_msg = (
        "Input data is in CSV format; converting to h5ad file at "
        f"{dst_path}"
    )

    if log is None:
        warnings.warn(warning_msg)
    else:
        log.warn(warning_msg)

    first_column_names = is_first_column_label(
        src_path=src_path
    )

    try:
        adata = anndata.io.read_csv(
            filename=src_path,
            delimiter=',',
            first_column_names=first_column_names)
    except Exception:
        full_msg = (
            "=======An error occurred when reading your CSV with anndata:\n"
            f"{traceback.format_exc()}\n"
            "Please confirm that your CSV is a table in which each "
            "row is a cell and each column is a gene."
        )
        raise RuntimeError(full_msg)

    adata.write_h5ad(dst_path)

    return (dst_path, True)


def is_first_column_label(src_path):
    """
    Accepts path to CSV file.
    Returns a boolean indicating whether or not the first
    column is to be treated as cell labels.
    """
    src_path = pathlib.Path(src_path)
    src_name = src_path.name

    if src_name.endswith('.csv'):
        open_fn = open
        mode = 'r'
        is_gzip = False
    elif src_name.endswith('.gz'):
        open_fn = gzip.open
        mode = 'rb'
        is_gzip = True
    else:
        msg = (
            "is_first_column_label unable to parse file "
            f"{src_name}; must end with .csv (or .gz if gzipped)"
        )
        raise RuntimeError(msg)

    with open_fn(src_path, mode) as src:
        header = src.readline()

    if is_gzip:
        header = header.decode()

    if is_first_header_column_blank(header_row=header):
        return True

    x_array = pd.read_csv(src_path).to_numpy()

    if not np.issubdtype(x_array[:, 0].dtype, np.number):
        return True

    if is_first_column_large(
            x_array=x_array,
            n_sig=3):
        return True

    if is_first_column_floats(x_array):
        return False

    if is_first_column_sequential(
            x_array=x_array):
        return True

    return False


def is_first_column_sequential(x_array):
    """
    Take the first column of a numpy array.
    Sort the values.
    If they are sequential when sorted, return True, else return False
    """
    sorted_col = np.sort(x_array[:, 0])
    delta = np.diff(sorted_col)
    return np.allclose(
        delta,
        np.ones(delta.shape),
        atol=0.0,
        rtol=1.0e-6
    )


def is_first_column_floats(x_array):
    """
    Return True if the first column of x_array (a numpy array)
    is floats. False if they are integers.
    Note: for purposes of this function, 1.0 is an integer.
    """
    if np.issubdtype(x_array.dtype, np.integer):
        return False
    col = x_array[:, 0]
    col_int = np.round(col)
    delta = np.abs(col-col_int)
    return delta.max() > 0.0


def is_first_column_large(x_array, n_sig=3):
    """
    Test if the first column of a numpy array is,
    in *all* rows, 3-sigma larger than the other
    non-zero values in the array.
    """
    first_col = x_array[:, 0]
    x_array = x_array[:, 1:]
    masked_x = np.ma.masked_array(
        x_array,
        mask=(x_array == 0.0)
    )
    mu = np.mean(masked_x, axis=1)
    std = np.std(masked_x, axis=1)
    return (
        first_col > (mu+3*std)
    ).all()


def is_first_header_column_blank(header_row):
    """
    Take the first row of a CSV file as a string.
    Return boolean indicating whether or not the first
    column (comma delimited) is blank.
    """
    header_row = header_row.replace('"', '')
    header_row = header_row.replace("'", "")
    header_row = header_row.replace(" ", "")
    header_params = header_row.split(',')
    if header_params[0] == '':
        return True
    return False
