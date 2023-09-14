import h5py
import numpy as np
import tempfile
import time

from cell_type_mapper.utils.multiprocessing_utils import (
    DummyLock)

from cell_type_mapper.utils.utils import (
    print_timing,
    merge_index_list,
    choose_int_dtype,
    mkstemp_clean,
    _clean_up)


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
     indptr) = _load_sparse(
                    indptr_spec=row_spec,
                    data=data,
                    indices=indices,
                    indptr=indptr)

    return _csr_to_dense(
                data=data,
                indices=indices,
                indptr=indptr,
                n_rows=row_spec[1]-row_spec[0],
                n_cols=n_cols)


def load_csc(
        col_spec,
        n_rows,
        data,
        indices,
        indptr):
    """
    Return a dense matrix from a subset of csc columns
    """

    (data,
     indices,
     indptr) = _load_sparse(
                    indptr_spec=col_spec,
                    data=data,
                    indices=indices,
                    indptr=indptr)

    return _csc_to_dense(
                data=data,
                indices=indices,
                indptr=indptr,
                n_cols=col_spec[1]-col_spec[0],
                n_rows=n_rows)


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
     indptr) = _load_sparse(
                    indptr_spec=row_spec,
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
    inverse_argsort = {sorted_dex[ii]: ii for ii in range(len(sorted_dex))}

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
         this_indptr) = _load_sparse(
                             indptr_spec=row_chunk,
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

    return final_data, final_indices, final_indptr


def _load_sparse(
        indptr_spec,
        data,
        indices,
        indptr):
    """
    Load a subset of rows/columns from a sparse matrix
    (probably in zarr format).

    Parameters
    ----------
    indptr_spec:
        A tuple of the form (indptr_min, indptr_max).
        For a CSR matrix, indptr_min/max will be rows.
        For a CSC matrix, indptr_min/max will be columns.

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
    these_ptrs = indptr[indptr_spec[0]:indptr_spec[1]+1]
    index0 = these_ptrs[0]
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
        these_cols = indices[indptr[iptr]:indptr[iptr+1]]
        n_cols = len(these_cols)
        result[iptr, these_cols] = data[data_idx:data_idx+n_cols]
        data_idx += n_cols

    return result


def _csc_to_dense(
        data,
        indices,
        indptr,
        n_rows,
        n_cols):
    """
    Return a dense matrix from a csc sparse matrix specification

    Parameters
    ----------
    data:
        The data matrix (as in scipy.sparse.csc_matrix().data)

    indices:
        The indices matrix (as in scipy.sparse.csc_matrix().indices)

    indptr:
        The indptr matrix (as in scipy.sparse.csc_matrix().indptr)

    n_rows:
        Number of rows in the dense matrix

    n_cols:
        Number of columns in the dense matrix
    """

    result = np.zeros((n_rows, n_cols),
                      dtype=data.dtype)

    data_idx = 0
    for iptr in range(len(indptr)-1):
        these_rows = indices[indptr[iptr]:indptr[iptr+1]]
        n_rows = len(these_rows)
        result[these_rows, iptr] = data[data_idx:data_idx+n_rows]
        data_idx += n_rows

    return result


def _cull_columns(
        col_spec,
        data,
        indices,
        indptr):
    """
    Return only the desired columns from a csr matrix
    """
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


def remap_csr_matrix(
        data_handle,
        indices_handle,
        indptr,
        new_indptr,
        new_row_order,
        writer_obj,
        flush_every=1000000,
        row_chunk=None,
        output_lock=None,
        process_name=None):
    """
    Given a CSR array and a re-arranged
    indptr array, write out the re-arrangeced
    CSR array.

    row_chunk is of the form (row_min, row_max)
    """
    data_buffer = np.zeros(flush_every, data_handle.dtype)
    indices_buffer = np.zeros(flush_every, int)
    output_0 = 0
    buffer_1 = 0

    new_to_old_row = dict()
    for ii, rr in enumerate(new_row_order):
        new_to_old_row[ii] = rr

    t0 = time.time()
    t_load = 0.0
    t_write = 0.0

    if row_chunk is not None:
        output_0 = new_indptr[row_chunk[0]]
    else:
        row_chunk = (0, len(indptr)-1)

    row_ct = 0
    for new_row in range(row_chunk[0], row_chunk[1], 1):
        old_row = new_to_old_row[new_row]
        i0 = indptr[old_row]
        i1 = indptr[old_row+1]
        _t0 = time.time()
        data_chunk = data_handle[i0:i1]
        indices_chunk = indices_handle[i0:i1]
        t_load += time.time()-_t0

        _t0 = time.time()
        (output_0,
         buffer_1) = _update_buffers(
                          data_buffer=data_buffer,
                          indices_buffer=indices_buffer,
                          writer_obj=writer_obj,
                          data_chunk=data_chunk,
                          indices_chunk=indices_chunk,
                          output_0=output_0,
                          buffer_1=buffer_1,
                          force_flush=False,
                          output_lock=output_lock)
        t_write += time.time()-_t0
        row_ct += 1

        if row_ct % 1000 == 0 or row_ct == 1:
            print_timing(
                t0=t0,
                tot_chunks=row_chunk[1]-row_chunk[0],
                i_chunk=row_ct,
                unit='hr',
                nametag=process_name)
            msg = f"spent {t_load/3600.0:.2e} hrs loading "
            msg += f"{t_write/3600.0:.2e} hrs writing"
            print(msg)

    _update_buffers(
          data_buffer=data_buffer,
          indices_buffer=indices_buffer,
          writer_obj=writer_obj,
          data_chunk=None,
          indices_chunk=None,
          output_0=output_0,
          buffer_1=buffer_1,
          force_flush=True,
          output_lock=output_lock)

    writer_obj.write_indptr(new_indptr=new_indptr)


def _update_buffers(
        writer_obj,
        data_buffer,
        indices_buffer,
        data_chunk,
        indices_chunk,
        output_0,
        buffer_1,
        force_flush=False,
        output_lock=None):
    """
    output_0 is the point in the output_handle where
    these buffers need to be stored

    data_buffer[:buffer_1] is valid at the beginning of this
    call.

    return updated values for output_0 and buffer_1
    """
    if force_flush:
        if buffer_1 > 0:
            output_1 = output_0+buffer_1
            writer_obj.write_data(
                i0=output_0,
                i1=output_1,
                data_chunk=data_buffer[:buffer_1],
                indices_chunk=indices_buffer[:buffer_1])
        if data_chunk is not None:
            writer_obj.write_data(
                i0=output_1,
                i1=output_1+data_chunk.shape[0],
                data_chunk=data_chunk,
                indices_chunk=indices_chunk)
        return None

    if buffer_1 + data_chunk.shape[0] < len(data_buffer):
        data_buffer[buffer_1:buffer_1+data_chunk.shape[0]] = data_chunk
        indices_buffer[buffer_1:buffer_1+data_chunk.shape[0]] = indices_chunk
        buffer_1 += data_chunk.shape[0]
        return (output_0,
                buffer_1)

    print("flushing")
    t0 = time.time()

    if output_lock is None:
        output_lock = DummyLock()

    with output_lock:
        if buffer_1 > 0:
            output_1 = output_0+buffer_1

            writer_obj.write_data(
                i0=output_0,
                i1=output_1,
                data_chunk=data_buffer[:buffer_1],
                indices_chunk=indices_buffer[:buffer_1])

            output_0 = output_1
            buffer_1 = 0

        if data_chunk is not None:
            output_1 = output_0 + data_chunk.shape[0]

            writer_obj.write_data(
                i0=output_0,
                i1=output_1,
                data_chunk=data_chunk,
                indices_chunk=indices_chunk)

            output_0 = output_1

    duration = (time.time()-t0)/3600.0
    print(f"flushing took {duration:.2e} hrs\n")
    return (output_0,
            buffer_1)


def downsample_indptr(
        indptr_old,
        indices_old,
        indptr_to_keep):
    """
    Downsample inpdtr and indices.

    Returns indptr_new, indices_new
    """

    ct_new = 0
    indptr_new = np.zeros(len(indptr_to_keep)+1, dtype=indptr_old.dtype)
    for ii, indptr in enumerate(indptr_to_keep):
        indptr_new[ii] = ct_new
        ct_new += indptr_old[indptr+1]-indptr_old[indptr]
    indptr_new[-1] = ct_new

    indices_new = np.zeros(ct_new, dtype=indices_old.dtype)
    for ii in range(len(indptr_to_keep)):
        src0 = indptr_old[indptr_to_keep[ii]]
        src1 = indptr_old[indptr_to_keep[ii]+1]
        dst0 = indptr_new[ii]
        dst1 = indptr_new[ii+1]
        indices_new[dst0:dst1] = indices_old[src0:src1]

    return indptr_new, indices_new


def mask_indptr_by_indices(
        indptr_old,
        indices_old,
        indices_map):
    """
    indices_map maps old index value to new index values
    (if an index has been dropped, it does not appear in
    indices_map)
    """
    bad_val = -999
    indices_old = np.array([
        indices_map[n]
        if n in indices_map else bad_val
        for n in indices_old])

    ct = (indices_old >= 0).sum()
    new_indices = np.zeros(ct, dtype=indices_old.dtype)
    new_indptr = np.zeros(len(indptr_old), dtype=indptr_old.dtype)
    ct = 0
    for ii in range(len(indptr_old)-1):
        src0 = indptr_old[ii]
        src1 = indptr_old[ii+1]
        valid = (indices_old[src0:src1] >= 0)
        n_valid = valid.sum()
        new_indptr[ii] = ct
        new_indices[ct:ct+n_valid] = np.sort(indices_old[src0:src1][valid])
        ct += n_valid
    new_indptr[-1] = len(new_indices)
    return (new_indptr, new_indices)


def amalgamate_sparse_array(
        src_rows,
        dst_path,
        sparse_grp=None,
        verbose=False,
        tmp_dir=None):
    """
    Take rows (or columns for csc matrices) from different
    sparse arrays stored in different HDF5 files and combine
    them into a single sparse array in a single HDF5 file.

    Parameters
    ----------
    src_rows:
        Ordered list of dicts. Each dict is
        {
            'path': /path/to/src/file
            'rows': [ordered list of rows/columns from that file]
        }

    dst_path:
        Path of file to be written

    sparse_grp:
        If not None, the prefix for the sparse array datasets
        in the source HDF5 files (i.e. 'X' for h5ad files)

    verbose:
        If True, issue print statements indicating the
        status of the copy

    tmp_dir:
        Directory where temporary files will be written
    """

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)

    try:
        _amalgamate_sparse_array(
            src_rows=src_rows,
            dst_path=dst_path,
            sparse_grp=sparse_grp,
            verbose=verbose,
            tmp_dir=tmp_dir)
    finally:
        if verbose:
            print(f"cleaning up {tmp_dir}")
        _clean_up(tmp_dir)


def _amalgamate_sparse_array(
        src_rows,
        dst_path,
        sparse_grp,
        verbose,
        tmp_dir):
    if sparse_grp is not None:
        indices_key = f'{sparse_grp}/indices'
        data_key = f'{sparse_grp}/data'
        indptr_key = f'{sparse_grp}/indptr'
    else:
        indices_key = 'indices'
        data_key = 'data'
        indptr_key = 'indtpr'

    ct_elements = 0
    ct_rows = 0
    max_index = -999
    data_is_int = True
    data_max = None
    data_min = None
    eps = 1.0e-20

    tmp_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='amalgamation_data_',
        suffix='.h5')

    src_dtypes = set()
    with h5py.File(tmp_path, 'w') as tmp_dst:
        for i_src, src_element in enumerate(src_rows):
            full_indices = []
            full_data = []
            full_indptr = []
            local_indptr_idx = 0
            src_path = src_element['path']
            if verbose:
                print(f"    running census on {src_path}")
            ct_rows += len(src_element['rows'])
            with h5py.File(src_path, 'r') as src:
                indices = src[indices_key][()]
                indptr = src[indptr_key][()]
                for idx in src_element['rows']:
                    full_indptr.append(local_indptr_idx)
                    i0 = indptr[idx]
                    i1 = indptr[idx+1]
                    this = indices[i0:i1]
                    full_indices.append(this)
                    ct_elements += len(this)
                    local_indptr_idx += len(this)
                    if len(this) > 0:
                        this_max = this.max()
                        if this_max > max_index:
                            max_index = this_max
                del indices
                data = src[data_key][()]
                src_dtypes.add(data.dtype)
                for idx in src_element['rows']:
                    i0 = indptr[idx]
                    i1 = indptr[idx+1]
                    this = data[i0:i1]
                    full_data.append(this)
                    if len(this) > 0:
                        delta = np.abs(np.round(this)-this)
                        if delta.max() > eps:
                            data_is_int = False
                        this_max = this.max()
                        this_min = this.min()
                        if data_max is None or this_max > data_max:
                            data_max = this_max
                        if data_min is None or this_min < data_min:
                            data_min = this_min

            if len(full_data) > 0:
                full_data = np.concatenate(full_data)
                full_indices = np.concatenate(full_indices)

            full_indptr.append(len(full_indices))

            tmp_data_key = f'src_{i_src}_data'
            tmp_indices_key = f'src_{i_src}_indices'
            tmp_indptr_key = f'src_{i_src}_indptr'
            tmp_dst.create_dataset(
                tmp_data_key,
                data=full_data)
            tmp_dst.create_dataset(
                tmp_indices_key,
                data=full_indices)
            tmp_dst.create_dataset(
                tmp_indptr_key,
                data=np.array(full_indptr))

    if verbose:
        print(f"    done with census; {ct_elements} non-zero elements")

    indices_dtype = choose_int_dtype((0, max_index))
    if data_is_int:
        data_dtype = choose_int_dtype((data_min, data_max))
    else:
        if len(src_dtypes) > 1:
            raise RuntimeError(
                "Cannot amalgamate; disparate data types\n"
                f"{list(src_dtypes)}")
        data_dtype = src_dtypes.pop()

    if verbose:
        print(f"    saving with dtype {data_dtype}")

    # oppen with mode='a' so that, if we are adding an 'X' element
    # to an anndata file, we don't blow away pre-existing obs/var
    # data structures
    with h5py.File(tmp_path, 'r') as tmp_src:
        with h5py.File(dst_path, 'a') as dst:
            if sparse_grp is not None:
                dst_handle = dst.create_group(sparse_grp)
            else:
                dst_handle = dst

            dst_handle.create_dataset(
                'data',
                shape=(ct_elements,),
                dtype=data_dtype,
                chunks=(min(ct_elements, 10000),))
            dst_handle.create_dataset(
                'indices',
                shape=(ct_elements,),
                dtype=indices_dtype,
                chunks=(min(ct_elements, 10000),))
            dst_handle.create_dataset(
                'indptr',
                shape=(ct_rows+1,),
                dtype=int,
                chunks=(min(ct_rows+1, 10000),))

            current_index = 0
            current_row = 0
            for i_src, src_element in enumerate(src_rows):
                indices = tmp_src[
                    f'src_{i_src}_indices'][()]
                indptr = tmp_src[
                   f'src_{i_src}_indptr'][()]
                data = tmp_src[
                    f'src_{i_src}_data'][()].astype(data_dtype)
                for idx in range(len(src_element['rows'])):
                    i0 = indptr[idx]
                    i1 = indptr[idx+1]
                    this_indices = indices[i0:i1]
                    this_data = data[i0:i1]
                    n = len(this_indices)
                    dst_handle['indices'][
                        current_index:current_index+n] = this_indices
                    dst_handle['data'][
                        current_index:current_index+n] = this_data
                    dst_handle['indptr'][current_row] = current_index
                    current_row += 1
                    current_index += n
                del indices
                del data
            dst_handle['indptr'][-1] = current_index
