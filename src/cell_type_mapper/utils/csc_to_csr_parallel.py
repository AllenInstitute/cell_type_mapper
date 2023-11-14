import h5py
import numpy as np
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean)


def transpose_sparse_matrix_on_disk_v2(
        indices_handle,
        indptr_handle,
        data_handle,
        n_indices,
        max_gb,
        output_path,
        verbose=False,
        tmp_dir=None):
    """
    n_indices is the number of unique indices values in original array
    """

    indices_dtype = int

    use_data = (data_handle is not None)

    n_raw_indices = indices_handle.shape[0]
    indices_gb = (4*n_raw_indices)/(1024**3)
    chunk_size = np.round(n_raw_indices*max_gb/indices_gb).astype(int)
    if chunk_size == 0:
        chunk_size = 10000
    if chunk_size > n_raw_indices:
        chunk_size = n_raw_indices

    indices_chunk_size = np.round(n_indices/32).astype(int)

    path_list = []
    for i0 in range(0, n_indices, indices_chunk_size):
        i1 = min(n_indices, i0+indices_chunk_size)

        tmp_path = pathlib.Path(
                mkstemp_clean(
                    dir=tmp_dir,
                    suffix='.h5',
                    prefix=f'transpose_{i0}_{i1}_'))

        _transpose_subset_of_indices(
            indices_handle=indices_handle,
            indptr_handle=indptr_handle,
            data_handle=data_handle,
            indices_minmax=(i0, i1),
            chunk_size=chunk_size,
            output_path=tmp_path)

        path_list.append(tmp_path)

    indices_size = 0
    indptr_size = 0
    for path in path_list:
        with h5py.File(path, 'r') as src:
            indices_size += src['indices'].shape[0]
            indptr_size += src['indptr'].shape[0]-1
    indptr_size += 1

    indptr_idx = 0
    indices_idx = 0
    with h5py.File(output_path, 'w') as dst:
        indices = dst.create_dataset(
            'indices',
            shape=(indices_size,),
            chunks=(min(indptr_size, 10000),),
            dtype=indices_dtype)
        indptr = dst.create_dataset(
            'indptr',
            shape=(indptr_size,),
            chunks=None,
            dtype=int)
        if use_data:
            data = dst.create_dataset(
                'data',
                shape=(indices_size,),
                chunks=(min(indptr_size, 10000),),
                dtype=data_handle.dtype)

        for path in path_list:
            with h5py.File(path, 'r') as src:
                src_indices = src['indices']
                src_indptr = src['indptr']
                if use_data:
                    src_data = src['data']
                src_n = src_indices.shape[0]
                src_n_ptr = src_indptr.shape[0]-1
                indptr[indptr_idx:indptr_idx+src_n_ptr] = src_indptr[:-1] + indices_idx

                dst0 = indices_idx
                for src0 in range(0, src_n, chunk_size):
                    src1 = min(src_n, src0+chunk_size)
                    dst1 = dst0 + (src1-src0)
                    indices[dst0:dst1] = src_indices[src0:src1]
                    if use_data:
                        data[dst0:dst1] = src_data[src0:src1]
                    dst0 = dst1

                indices_idx += src_n
                indptr_idx += src_n_ptr
            path.unlink()
        indptr[-1] = indices_idx

def _transpose_subset_of_indices(
        indices_handle,
        indptr_handle,
        data_handle,
        indices_minmax,
        chunk_size,
        output_path):


    use_data = (data_handle is not None)

    output_dict = dict()
    n_indices = indices_handle.shape[0]
    indptr = indptr_handle[()]
    for i0 in range(0, n_indices, chunk_size):
        i1 = min(n_indices, i0+chunk_size)
        if use_data:
            data_chunk = data_handle[i0:i1]
        else:
            data_chunk = None
        indices_chunk = indices_handle[i0:i1]
        _grab_indices_from_chunk(
            indptr=indptr,
            indptr_0=0,
            indices_chunk=indices_chunk,
            data_chunk=data_chunk,
            indices_minmax=indices_minmax,
            indices_position=(i0, i1),
            output_dict=output_dict)

    indptr = []
    indices = []
    data = []
    indptr_ct = 0
    for i_indptr in range(indices_minmax[0], indices_minmax[1], 1):
        indptr.append(indptr_ct)
        if i_indptr not in output_dict:
            continue
        this = output_dict.pop(i_indptr)
        n = 0
        for row in this['indices']:
            n += len(row)
        indptr_ct += n
        indices += this['indices']
        if use_data:
            data += this['data']

    indptr.append(indptr_ct)

    if len(indices) > 0:
        indices = np.concatenate(indices)
    else:
        indices = np.array([])

    if use_data:
        if len(data) > 0:
            data = np.concatenate(data)
        else:
            data = np.array([])

    with h5py.File(output_path, 'w') as dst:
        dst.create_dataset(
            'indices_minmax',
            data=np.array(indices_minmax))
        dst.create_dataset(
            'indices',
            data=indices)
        dst.create_dataset(
            'indptr',
            data=indptr)
        if use_data:
            dst.create_dataset(
                'data',
                data=data)


def _grab_indices_from_chunk(
        indptr,
        indptr_0,
        indices_chunk,
        data_chunk,
        indices_minmax,
        indices_position,
        output_dict):
    """
    Parameters
    ----------
    inpdtr:
        chunk of original indptr array that is relevant
    indptr_0:
        offset in indptr_idx (in case indptr is not the raw
        indptr)
    indices_chunk:
        the chunk of indices to be processed now
    data_chunk:
        the chunk of the data array being transposed
        (can be None)
    indices_minmax:
        tuple indicating min, max of indices values being
        considered by this worker
    indices_position:
        tuple indicating (i0, i1) of indices_chunk (i.e. indices
        chunk is global_indices[i0:i1])
    output_dict:
        dict mapping value in original indices to array of original
        indptr values

    Notes
    -----
    This function will scan through the chunk of indices, grabbing
    the desired indices and populating a dict that maps indices
    to indptr (since, in the transposed array, indices<->indptr)

    output_dict[ii] will be a list of arrays indicating the
    "column" values for that index.
    """
    indices_idx = np.arange(indices_position[0], indices_position[1], dtype=int)
    valid = np.ones(indices_chunk.shape, dtype=bool)
    valid[indices_chunk<indices_minmax[0]] = False
    valid[indices_chunk>=indices_minmax[1]] = False
    indices_idx = indices_idx[valid]
    indices_chunk = indices_chunk[valid]
    if data_chunk is not None:
         data_chunk = data_chunk[valid]

    indptr_idx = np.zeros(indices_idx.shape, dtype=int)
    for ii in range(len(indptr)-1):
        row = ii+indptr_0
        idx = indptr[ii]
        valid = np.logical_and(indices_idx>=idx, indices_idx<indptr[ii+1])
        indptr_idx[valid] = row

    for unq in np.unique(indices_chunk):
        valid = (indices_chunk==unq)
        this = indptr_idx[valid]
        if unq not in output_dict:
            output_dict[unq] = {'indices': [], 'data': []}
        output_dict[unq]['indices'].append(this)
        if data_chunk is not None:
            output_dict[unq]['data'].append(data_chunk[valid])
