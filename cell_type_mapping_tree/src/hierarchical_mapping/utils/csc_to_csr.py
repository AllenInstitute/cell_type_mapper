import h5py
import numpy as np


def csc_to_csr_on_disk(
        csc_group,
        csr_path,
        array_shape,
        load_chunk_size=500000):
    """
    Convert a large csc matrix to an on-disk
    csr matrix at the specified location

    csc_group is the group within an on-disk
    HDF file file containing the
    'data'
    'indices'
    'indptr'
    arrays for the CSC matrix
    """

    n_cols = array_shape[1]
    log2_n_cols = np.log2(n_cols)
    if log2_n_cols < 8:
        index_dtype = np.uint8
    elif log2_n_cols < 16:
        index_dtype = np.uint16
    elif log2_n_cols < 32:
        index_dtype = np.uint32
    else:
        index_dtype = np.uint64

    n_non_zero = csc_group['data'].shape[0]
    with h5py.File(csr_path, 'w') as dst:
        dst.create_dataset(
            'data',
            shape=n_non_zero,
            dtype=csc_group['data'].dtype,
            chunks=(min(n_non_zero, array_shape[1])))
        dst.create_dataset(
            'indices',
            shape=n_non_zero,
            dtype=index_dtype,
            chunks=(min(n_non_zero, array_shape[1])))

    print(f"created empty csr matrix at {csr_path}")

    cumulative_count = np.zeros(array_shape[0], dtype=int)
    for i0 in range(0, n_non_zero, load_chunk_size):
        i1 = min(n_non_zero, i0+load_chunk_size)
        chunk = csc_group['indices'][i0:i1]
        unq_val, unq_ct = np.unique(chunk, return_counts=True)
        cumulative_count[unq_val] += unq_ct
    csr_indptr = np.cumsum(cumulative_count)
    csr_indptr = np.concatenate([np.array([0], dtype=int), csr_indptr])
    next_idx = np.copy(csr_indptr)

    csc_indptr = csc_group['indptr'][()]
    with h5py.File(csr_path, 'a') as dst:
        dst.create_dataset('indptr', data=csr_indptr)
        for i0 in range(0, n_non_zero, load_chunk_size):
            i1 = min(n_non_zero, i0+load_chunk_size)
            i_chunk = np.arange(i0, i1, dtype=int)
            col_chunk = np.searchsorted(csc_indptr, i_chunk, side='right')
            col_chunk -= 1

            data_chunk = csc_group['data'][i0:i1]
            row_chunk = csc_group['indices'][i0:i1]
            sorted_dex = np.argsort(row_chunk)

            row_chunk = row_chunk[sorted_dex]
            data_chunk = data_chunk[sorted_dex]
            col_chunk = col_chunk[sorted_dex]

            unq_val_arr, unq_ct_arr = np.unique(row_chunk, return_counts=True)
            for unq_val, unq_ct in zip(unq_val_arr, unq_ct_arr):
                j0 = np.searchsorted(row_chunk, unq_val, side='left')
                d0 = next_idx[unq_val]
                this_data = data_chunk[j0:j0+unq_ct]
                this_index = col_chunk[j0:j0+unq_ct]

                col_sorted_dex = np.argsort(this_index)

                this_index = this_index[col_sorted_dex]
                this_data = this_data[col_sorted_dex]

                dst['data'][d0:d0+unq_ct] = this_data
                dst['indices'][d0:d0+unq_ct] = this_index
                next_idx[unq_val] += unq_ct
