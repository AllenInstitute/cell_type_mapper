import numpy as np
import time

from hierarchical_mapping.utils.utils import merge_index_list


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


def _load_disjoint_csr(
        row_index_list,
        data,
        indices,
        indptr):
    """
    Load a csr matrix from a not necessarily contiguous
    set of row indexes.
    """
    row_index_list = np.array(row_index_list)
    sorted_dex = np.argsort(row_index_list)
    inverse_argsort = {sorted_dex[ii]:ii for ii in range(len(sorted_dex))}

    row_index_list = row_index_list[sorted_dex]

    row_chunk_list = merge_index_list(row_index_list)
    data_list = []
    indices_list = []
    indptr_list = []
    t_load = 0.0
    for row_chunk in row_chunk_list:
        t0 = time.time()
        (this_data,
         this_indices,
         this_indptr) = _load_csr(
                             row_spec=row_chunk,
                             data=data,
                             indices=indices,
                             indptr=indptr)
        t_load += time.time()-t0

        data_list.append(this_data)
        indices_list.append(this_indices)
        indptr_list.append(this_indptr)

    (merged_data,
     merged_indices,
     merged_indptr) = merge_csr(
                         data_list=data_list,
                         indices_list=indices_list,
                         indptr_list=indptr_list)

    # undo sorting
    final_data = np.zeros(merged_data.shape, dtype=merged_data.dtype)
    final_indices = np.zeros(merged_indices.shape, dtype=merged_indices.dtype)
    final_indptr = np.zeros(merged_indptr.shape, dtype=merged_indptr.dtype)

    data_ct = 0
    for ii in range(len(row_index_list)):
        new_position = inverse_argsort[ii]
        indptr0 = merged_indptr[new_position]
        indptr1 = merged_indptr[new_position+1]
        n = indptr1-indptr0
        final_data[data_ct:data_ct+n] = merged_data[indptr0:indptr1]
        final_indices[data_ct:data_ct+n] = merged_indices[indptr0:indptr1]
        final_indptr[ii] = data_ct
        data_ct += n
    final_indptr[-1] = len(final_data)

    print(f"time spent loading disjoint {t_load:.2e} seconds")
    return final_data, final_indices, final_indptr


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
    t0 = time.time()
    #print("loading csr")
    these_ptrs = indptr[row_spec[0]:row_spec[1]+1]
    #print(f"got indptr {time.time()-t0:.2e}")
    index0= these_ptrs[0]
    index1 = these_ptrs[-1]
    these_indices = indices[index0:index1]
    #print(f"got indices {time.time()-t0:.2e}")
    this_data = data[index0:index1]
    #print(f"done loading csr {time.time()-t0:.2e}")
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
        these_cols = indices[indptr[iptr]:indptr[iptr+1]]
        n_cols = len(these_cols)
        result[iptr, these_cols] = data[data_idx:data_idx+n_cols]
        data_idx += n_cols

    return result


def _cull_columns(
        col_spec,
        data,
        indices,
        indptr):
    """
    Return only the desired columns from a csr matrix
    """
    #print("culling columns")
    valid_idx = np.where(
            np.logical_and(
                indices >= col_spec[0],
                indices < col_spec[1]))[0]

    new_data = data[valid_idx]
    new_indices = indices[valid_idx]-col_spec[0]

    this_row = 0
    new_indptr = np.zeros(len(indptr), dtype=int)
    new_indptr[0]
    for new_idx, this_idx in enumerate(valid_idx):
        while this_idx >= indptr[this_row+1]:
            this_row += 1
            new_indptr[this_row] = new_idx

    new_indptr[this_row+1:] = len(new_data)

    #print("done culling columns")
    return (new_data,
            new_indices,
            new_indptr)


def merge_csr(
        data_list,
        indices_list,
        indptr_list):
    """
    Merge multiple CSR matrices into one

    Parameters
    ----------
    data_list:
        List of the distinct 'data' arrays from
        the CSR matrices

    indices_list:
        List of the distinct 'indices' arrays from
        the CSR matrices

    indptr_list:
        List of the distinct 'indptr' arrays from
        the CSR matrices

    Returns
    -------
    data:
        merged 'data' array for the final CSR matrix

    indices:
        merged 'indices' array for the final CSR matrix

    indptr:
        merged 'indptr' array for the final CSR matrix
    """
    n_data = 0
    for d in data_list:
        n_data += len(d)
    n_indptr = 0
    for i in indptr_list:
        n_indptr += len(i)-1
    n_indptr += 1

    data = np.zeros(n_data, dtype=data_list[0].dtype)
    indices = np.zeros(n_data, dtype=int)
    indptr = np.zeros(n_indptr, dtype=int)

    i0 = 0
    ptr0 = 0
    for this_data, this_indices, this_indptr in zip(data_list,
                                                    indices_list,
                                                    indptr_list):

        (data,
         indices,
         indptr,
         idx1,
         ptr1) = _merge_csr_chunk(
                 data_in=this_data,
                 indices_in=this_indices,
                 indptr_in=this_indptr,
                 data=data,
                 indices=indices,
                 indptr=indptr,
                 idx0=i0,
                 ptr0=ptr0)

        i0 = idx1
        ptr0 = ptr1

    indptr[-1] = len(data)
    return data, indices, indptr


def _merge_csr_chunk(
        data_in,
        indices_in,
        indptr_in,
        data,
        indices,
        indptr,
        idx0,
        ptr0):
    idx1 = idx0 + len(data_in)
    data[idx0:idx1] = data_in
    indices[idx0: idx1] = indices_in
    ptr1 = ptr0 + len(indptr_in)-1
    indptr[ptr0:ptr1] = indptr_in[:-1] + idx0

    return data, indices, indptr, idx1, ptr1


def precompute_indptr(
        indptr_in,
        row_order):
    """
    Take an indptr array from a CSR array and compute
    the new indptr array you get when you re-order the
    rows as specified in row_order
    """

    new_indptr = np.zeros(len(indptr_in), dtype=int)
    ct = 0
    for new_idx, row in enumerate(row_order):
        span = indptr_in[row+1]-indptr_in[row]
        new_indptr[new_idx] = ct
        ct += span
    new_indptr[-1] = indptr_in[-1]
    return new_indptr
