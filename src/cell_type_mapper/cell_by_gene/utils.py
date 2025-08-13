import numpy as np

from cell_type_mapper.utils.torch_utils import (
    use_torch)

try:
    import torch
except ImportError:
    pass


def convert_to_cpm(
        data,
        counts_per=1.0e6):
    """
    Convert a cell-by-gene array from raw counts to
    counts per million.

    Parameters
    ----------
    data:
        A numpy array of cell-by-gene data (each row is a cell;
        each column is a gene)
    counts_per:
        Factor to normalize cell counts to (i.e. the "million"
        in "counts per million" or the "thousand" in "counts
        per thousand"; defaults to one million)

    Returns
    -------
    cpm_data:
        data converted to "counts per million"
    """
    if use_torch():
        if torch.is_tensor(data):
            row_sums = torch.sum(data, axis=1)
            denom = torch.where(row_sums > 0.0, row_sums, 1.)
            cpm = torch.t(data)/denom
            cpm = counts_per*cpm
            return torch.t(cpm)

    row_sums = np.sum(data, axis=1)
    denom = np.where(row_sums > 0.0, row_sums, 1.)
    cpm = data.transpose()/denom
    cpm = counts_per*cpm
    return cpm.transpose()
