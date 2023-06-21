import numpy as np

from hierarchical_mapping.utils.torch_utils import(
    is_cuda_available,
    use_torch)

try:
    import torch
except ImportError:
    pass
    
    
def convert_to_cpm(
        data):
    """
    Convert a cell-by-gene array from raw counts to
    counts per million.

    Parameters
    ----------
    data:
        A numpy array of cell-by-gene data (each row is a cell;
        each column is a gene)

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
            cpm = 1.0e6*cpm
            return torch.t(cpm)

    row_sums = np.sum(data, axis=1)
    denom = np.where(row_sums > 0.0, row_sums, 1.)
    cpm = data.transpose()/denom
    cpm = 1.0e6*cpm
    return cpm.transpose()
