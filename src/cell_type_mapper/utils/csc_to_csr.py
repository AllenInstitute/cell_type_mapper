import h5py
import numpy as np
import pathlib
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)


def csc_to_csr_on_disk(
        csc_group,
        csr_path,
        array_shape,
        max_gb=15,
        use_data_array=True):
    """
    Convert a large csc matrix to an on-disk
    csr matrix at the specified location

    csc_group is the group within an on-disk
    HDF file file containing the
    'data'
    'indices'
    'indptr'
    arrays for the CSC matrix

    csr_path is the path to the HDF5 file that will get written

    array_shape is a tuple indicating the shape of the dense array
    we are converting

    if use_data_array is False, then there is no data array and
    we are just transposing the indices and indptr arrays
    """

    if use_data_array:
        data_handle = csc_group['data']
    else:
        data_handle = None

    transpose_sparse_matrix_on_disk(
        indices_handle=csc_group['indices'],
        indptr_handle=csc_group['indptr'],
        data_handle=data_handle,
        n_indices=array_shape[0],
        max_gb=max_gb,
        output_path=csr_path)


def transpose_by_way_of_disk(
        indices,
        indptr,
        n_indices,
        max_gb=10,
        tmp_dir=None):
    """
    Transpose a sparse matrix by writing it to disk
    in an h5ad file and then using transpose_sparse_matrix_on_disk.

    Return indptr, indices for the transposed matrix.

    (presently ignore 'data' since that's how we are using this tool)

    (n_indices is the size of the array dimension encoded by the
    old indices value)
    """
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir,
            prefix='transposing_sparse_matrix_'))

    src_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            prefix='src_',
            suffix='.h5'))

    dst_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            prefix='dst_',
            suffix='.h5'))

    with h5py.File(src_path, 'w') as dst:
        chunks = min(len(indices), 10000)
        if chunks == 0:
            chunks = None
        dst.create_dataset(
            'indices',
            data=indices,
            chunks=chunks)
        dst.create_dataset(
            'indptr',
            data=indptr)

    with h5py.File(src_path, 'r') as src:
        transpose_sparse_matrix_on_disk(
            indices_handle=src['indices'],
            indptr_handle=src['indptr'],
            data_handle=None,
            n_indices=n_indices,
            max_gb=max_gb,
            output_path=dst_path,
            verbose=False)
    with h5py.File(dst_path, 'r') as src:
        indptr = src['indptr'][()]
        indices = src['indices'][()]

    _clean_up(tmp_dir)
    return indptr, indices


def transpose_sparse_matrix_on_disk(
        indices_handle,
        indptr_handle,
        data_handle,
        n_indices,
        max_gb,
        output_path,
        verbose=True):

    use_data_array = (data_handle is not None)

    # to account for phantom overhead in numpy
    max_gb = 0.8*max_gb

    n_non_zero = indices_handle.shape[0]

    n_indptr = indptr_handle.shape[0]-1
    col_dtype = _get_uint_dtype(n_indptr)
    row_dtype = _get_uint_dtype(n_indices)

    if use_data_array:
        data_dtype = data_handle.dtype

    with h5py.File(output_path, 'w') as dst:

        if use_data_array:
            dst.create_dataset(
                'data',
                shape=n_non_zero,
                dtype=data_dtype,
                chunks=(min(n_non_zero, n_indptr),))

        if n_non_zero > 0:
            chunks = (min(n_non_zero, n_indptr),)
        else:
            chunks = None
        dst.create_dataset(
            'indices',
            shape=n_non_zero,
            dtype=col_dtype,
            chunks=chunks)

    csr_indptr = _calculate_csr_indptr(
        indices_handle=indices_handle,
        n_indices=n_indices,
        n_non_zero=n_non_zero,
        max_gb=max_gb,
        verbose=verbose)

    with h5py.File(output_path, 'a') as dst:
        dst.create_dataset(
            'indptr', data=csr_indptr)

    next_idx = np.copy(csr_indptr)
    csc_indptr = indptr_handle[()]
    row_group = indices_handle

    if use_data_array:
        data_group = data_handle
        data_bytes = _get_bytes_for_type(data_dtype)
    else:
        data_bytes = 0

    col_bytes = _get_bytes_for_type(col_dtype)
    row_bytes = _get_bytes_for_type(row_dtype)
    dex_bytes = 8

    max_load_gb = max_gb / 3
    max_el_gb = max_gb - max_load_gb

    load_bytes = int(data_bytes+col_bytes+row_bytes+dex_bytes)
    load_chunk_size = np.round(max_load_gb*1024**3).astype(int)//(load_bytes)

    el_bytes = int(data_bytes+col_bytes)
    elements_at_a_time = np.round(max_el_gb*1024**3).astype(int)//el_bytes

    load_chunk_size = max(100, load_chunk_size)
    elements_at_a_time = max(100, elements_at_a_time)

    r0 = 0

    while True:
        r1 = None
        e0 = csr_indptr[r0]
        for candidate in range(r0+1, len(csr_indptr), 1):
            e1 = csr_indptr[candidate]
            if e1-e0 >= elements_at_a_time or candidate == len(csr_indptr)-1:
                r1 = candidate
                break

        if r1 is None:
            break

        d0 = csr_indptr[r0]
        d1 = csr_indptr[r1]

        if use_data_array:
            data_buffer = np.zeros(d1-d0, dtype=data_dtype)

        index_buffer = np.zeros(d1-d0, dtype=col_dtype)

        for i0 in range(0, n_non_zero, load_chunk_size):
            i1 = min(n_non_zero, i0+load_chunk_size)

            row_chunk = row_group[i0:i1].astype(row_dtype)
            sorted_dex = np.argsort(row_chunk)
            row_chunk = row_chunk[sorted_dex]

            unq_val_arr, unq_ct_arr = np.unique(
                    row_chunk, return_counts=True)

            valid_dex = np.where(
                np.logical_and(
                    unq_val_arr >= r0,
                    unq_val_arr < r1))[0]

            if len(valid_dex) == 0:
                continue

            col_chunk = np.searchsorted(
                csc_indptr,
                np.arange(i0, i1, dtype=int),
                side='right')
            col_chunk -= 1
            col_chunk = col_chunk[sorted_dex]

            if use_data_array:
                data_chunk = data_group[i0:i1]
                data_chunk = data_chunk[sorted_dex]

            del sorted_dex

            unq_val_arr = unq_val_arr[valid_dex]
            unq_ct_arr = unq_ct_arr[valid_dex]

            for unq_val, unq_ct in zip(unq_val_arr, unq_ct_arr):
                j0 = np.searchsorted(row_chunk, unq_val, side='left')

                if use_data_array:
                    this_data = data_chunk[j0:j0+unq_ct]

                this_index = col_chunk[j0:j0+unq_ct]

                col_sorted_dex = np.argsort(this_index)

                this_index = this_index[col_sorted_dex]

                if use_data_array:
                    this_data = this_data[col_sorted_dex]

                buffer_0 = next_idx[unq_val]-d0
                buffer_1 = buffer_0+unq_ct

                if use_data_array:
                    data_buffer[buffer_0:buffer_1] = this_data

                index_buffer[buffer_0:buffer_1] = this_index
                next_idx[unq_val] += unq_ct

        with h5py.File(output_path, 'a') as dst:
            if use_data_array:
                dst['data'][d0:d1] = data_buffer
            dst['indices'][d0:d1] = index_buffer
        r0 = r1


def _calculate_csr_indptr(
        indices_handle,
        n_indices,
        n_non_zero,
        max_gb,
        verbose=True):

    bytes_per = _get_bytes_for_type(indices_handle.dtype)
    load_chunk_size = np.round(max_gb*1024**3).astype(int)//bytes_per
    load_chunk_size = load_chunk_size//2

    load_chunk_size = max(100, load_chunk_size)

    cumulative_count = np.zeros(n_indices, dtype=int)

    for i0 in range(0, n_non_zero, load_chunk_size):
        i1 = min(n_non_zero, i0+load_chunk_size)
        chunk = indices_handle[i0:i1]
        unq_val, unq_ct = np.unique(chunk, return_counts=True)
        cumulative_count[unq_val] += unq_ct
    csr_indptr = np.cumsum(cumulative_count)
    csr_indptr = np.concatenate([np.array([0], dtype=int), csr_indptr])
    return csr_indptr


def _get_uint_dtype(max_value):
    result = None
    for candidate in (np.uint8, np.uint16, np.uint32, np.uint):
        if max_value < np.iinfo(candidate).max:
            result = candidate
            break
    if result is None:
        raise RuntimeError(
            f"Could not find valid uint type for max_value {max_value}")
    return result


def _get_bytes_for_type(this_dtype):
    if np.issubdtype(this_dtype, np.integer):
        return np.iinfo(this_dtype).bits//8
    return np.finfo(this_dtype).bits//8
