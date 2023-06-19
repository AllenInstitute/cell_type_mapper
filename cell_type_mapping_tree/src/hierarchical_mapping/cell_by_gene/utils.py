import numpy as np

try:
    TORCH_AVAILABLE = False
    import torch  # type: ignore
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    
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
    if TORCH_AVAILABLE:
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
