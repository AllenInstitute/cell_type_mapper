import numpy as np


def load_csr(
        row_spec,
        n_cols,
        data,
        indices,
        indptr):
    """
    Return a dense matrix from a subset of csr rows
    """

    (data,
     indices,
     indptr) = _load_csr(
                    row_spec=row_spec,
                    data=data,
                    indices=indices,
                    indptr=indptr)

    result = np.zeros((row_spec[1]-row_spec[0], n_cols),
                      dtype=data.dtype)

    data_idx = 0
    for iptr in range(len(indptr)-1):
        for icol in indices[indptr[iptr]:indptr[iptr+1]]:
            result[iptr, icol] = data[data_idx]
            data_idx += 1

    return result


def _load_csr(
        row_spec,
        data,
        indices,
        indptr):
    """
    Load a subset of rows from a matrix stored as a
    csr matrix (probably in zarr format).

    Parameters
    ----------
    row_spec:
        A tuple of the form (row_min, row_max)

    data:
        The data matrix (as in scipy.sparse.csr_matrix().data)
    
    indices:
        The indices matrix (as in scipy.sparse.csr_matrix().indices)
    
    indptr:
        The indptr matrix (as in scipy.sparse.csr_matrix().indptr)

    Returns
    -------
    The appropriate slices of data, indices, indptr 
    """

    index0 = indptr[row_spec[0]]
    index1 = indptr[row_spec[1]]
    these_ptrs = indptr[row_spec[0]:row_spec[1]+1]
    index0= these_ptrs[0]
    index1 = these_ptrs[-1]
    these_indices = indices[index0:index1]
    this_data = data[index0:index1]
    return this_data, these_indices, these_ptrs-these_ptrs.min()
    
