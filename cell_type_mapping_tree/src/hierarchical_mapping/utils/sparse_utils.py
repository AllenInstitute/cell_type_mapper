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

    return _csr_to_dense(
                data=data,
                indices=indices,
                indptr=indptr,
                n_rows=row_spec[1]-row_spec[0],
                n_cols=n_cols)


def load_csr_chunk(
        row_spec,
        col_spec,
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

    (data,
     indices,
     indptr) = _cull_columns(
                     col_spec=col_spec,
                     data=data,
                     indices=indices,
                     indptr=indptr)

    return _csr_to_dense(
                data=data,
                indices=indices,
                indptr=indptr,
                n_rows=row_spec[1]-row_spec[0],
                n_cols=col_spec[1]-col_spec[0])


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


def _csr_to_dense(
        data,
        indices,
        indptr,
        n_rows,
        n_cols):
    """
    Return a dense matrix from a csr sparse matrix specification

    Parameters
    ----------
    data:
        The data matrix (as in scipy.sparse.csr_matrix().data)

    indices:
        The indices matrix (as in scipy.sparse.csr_matrix().indices)

    indptr:
        The indptr matrix (as in scipy.sparse.csr_matrix().indptr)

    n_rows:
        Number of rows in the dense matrix

    n_cols:
        Number of columns in the dense matrix
    """

    result = np.zeros((n_rows, n_cols),
                      dtype=data.dtype)

    data_idx = 0
    for iptr in range(len(indptr)-1):
        for icol in indices[indptr[iptr]:indptr[iptr+1]]:
            result[iptr, icol] = data[data_idx]
            data_idx += 1

    return result


def _cull_columns(
        col_spec,
        data,
        indices,
        indptr):
    """
    Return only the desired columns from a csr matrix
    """
    import time
    t0 = time.time()
    new_data = []
    new_indices = []
    new_indptr = []

    indices = np.array(indices)
    valid_columns = np.logical_and(
            indices >= col_spec[0],
            indices < col_spec[1])

    valid_idx = np.arange(len(indices), dtype=int)[valid_columns]
    new_data = data[valid_columns]
    new_indices = indices[valid_columns]-col_spec[0]

    this_row = 0
    print(f"ready for new indptr in {time.time()-t0:.2e}")
    new_indptr = np.zeros(len(indptr), dtype=int)
    new_indptr[0]
    for new_idx, this_idx in enumerate(valid_idx):

        while this_idx >= indptr[this_row+1]:
            this_row += 1
            new_indptr[this_row] = new_idx

    new_indptr[this_row+1:] = len(new_data)
    print(f"total took {time.time()-t0:.2e}")

    return (new_data,
            new_indices,
            new_indptr)
