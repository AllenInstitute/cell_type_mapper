import h5py
import multiprocessing
import numpy as np
import pathlib
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list)


def transpose_sparse_matrix_on_disk_v2(
        h5_path,
        indices_tag,
        indptr_tag,
        data_tag,
        n_indices,
        max_gb,
        output_path,
        verbose=False,
        tmp_dir=None,
        n_processors=4):
    """
    n_indices is the number of unique indices values in original array
    """

    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir,
        prefix='transposition')

    try:
        _transpose_sparse_matrix_on_disk_v2(
            h5_path=h5_path,
            indices_tag=indices_tag,
            indptr_tag=indptr_tag,
            data_tag=data_tag,
            n_indices=n_indices,
            max_gb=max_gb,
            output_path=output_path,
            verbose=verbose,
            tmp_dir=tmp_dir,
            n_processors=n_processors)
    finally:
        _clean_up(tmp_dir)


def _transpose_sparse_matrix_on_disk_v2(
        h5_path,
        indices_tag,
        indptr_tag,
        data_tag,
        n_indices,
        max_gb,
        output_path,
        verbose=False,
        tmp_dir=None,
        n_processors=4):
    indices_dtype = int

    use_data = (data_tag is not None)

    with h5py.File(h5_path, 'r') as src:
        n_raw_indices = src[indices_tag].shape[0]
        if use_data:
            data_dtype = src[data_tag].dtype

    indices_gb = (4*n_raw_indices)/(1024**3)
    chunk_size = np.round(n_raw_indices*max_gb/indices_gb).astype(int)
    chunk_size = np.round(chunk_size/n_processors).astype(int)
    if chunk_size == 0:
        chunk_size = 10000
    if chunk_size > n_raw_indices:
        chunk_size = n_raw_indices

    indices_chunk_size = np.round(n_indices/n_processors).astype(int)

    path_list = []
    process_list = []
    for i0 in range(0, n_indices, indices_chunk_size):
        i1 = min(n_indices, i0+indices_chunk_size)

        tmp_path = pathlib.Path(
                mkstemp_clean(
                    dir=tmp_dir,
                    suffix='.h5',
                    prefix=f'transpose_{i0}_{i1}_'))

        p = multiprocessing.Process(
            target=_transpose_subset_of_indices,
            kwargs={
                'h5_path': h5_path,
                'indices_tag': indices_tag,
                'indptr_tag': indptr_tag,
                'data_tag': data_tag,
                'indices_minmax': (i0, i1),
                'chunk_size': chunk_size,
                'output_path': tmp_path
            })

        p.start()
        process_list.append(p)
        path_list.append(tmp_path)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)
    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    indices_size = 0
    indptr_size = 0
    for path in path_list:
        with h5py.File(path, 'r') as src:
            indices_size += src['indices'].shape[0]
            indptr_size += src['indptr'].shape[0]-1
    indptr_size += 1

    t0 = time.time()
    indptr_idx = 0
    indices_idx = 0
    with h5py.File(output_path, 'w') as dst:
        indices = dst.create_dataset(
            'indices',
            shape=(indices_size,),
            chunks=(min(indptr_size, 1000000),),
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
                chunks=(min(indptr_size, 1000000),),
                dtype=data_dtype)

        for path in path_list:
            with h5py.File(path, 'r') as src:
                src_indices = src['indices']
                src_indptr = src['indptr']
                if use_data:
                    src_data = src['data']
                src_n = src_indices.shape[0]
                src_n_ptr = src_indptr.shape[0]-1
                indptr[indptr_idx:indptr_idx+src_n_ptr] = (src_indptr[:-1]
                                                           + indices_idx)

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
    dur = time.time()-t0
    print(f'joining took {dur:2e} seconds')


def _transpose_subset_of_indices(
        h5_path,
        indices_tag,
        indptr_tag,
        data_tag,
        indices_minmax,
        chunk_size,
        output_path):

    use_data = (data_tag is not None)

    with h5py.File(h5_path, 'r', swmr=True) as src:
        indices_handle = src[indices_tag]
        indptr_handle = src[indptr_tag]
        if use_data:
            data_handle = src[data_tag]
        else:
            data_handle = None

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
    indices_idx = np.arange(indices_position[0],
                            indices_position[1],
                            dtype=int)
    full_valid = np.ones(indices_chunk.shape, dtype=bool)
    full_valid[indices_chunk < indices_minmax[0]] = False
    full_valid[indices_chunk >= indices_minmax[1]] = False
    indices_chunk = indices_chunk[full_valid]
    masked_indices_idx = np.copy(indices_idx[full_valid])
    if data_chunk is not None:
        data_chunk = data_chunk[full_valid]

    indptr_idx = np.zeros(masked_indices_idx.shape, dtype=int)
    indptr_pos = 0
    for ii in range(len(indptr)-1):
        row = ii+indptr_0
        original0 = indptr[ii] - indices_position[0]
        original1 = indptr[ii+1] - indices_position[0]
        if original1 < 0:
            continue
        if original0 < 0:
            original0 = 0
        mask = full_valid[original0:original1]
        n_valid = mask.sum()
        indptr_idx[indptr_pos:indptr_pos+n_valid] = row
        indptr_pos += n_valid

    indices_idx = masked_indices_idx

    del full_valid
    del indices_idx

    if len(indptr_idx) == 0:
        return dict()

    # sort first by indices then by indptr
    max_indptr = indptr_idx.max()
    to_sort = (indices_chunk.astype(np.int64)*(max_indptr+1)
               + indptr_idx.astype(np.int64))
    sorted_dex = np.argsort(to_sort)

    del to_sort

    indices_chunk = indices_chunk[sorted_dex]
    indptr_idx = indptr_idx[sorted_dex]
    if data_chunk is not None:
        data_chunk = data_chunk[sorted_dex]

    delta_indices = np.diff(indices_chunk)

    # these will be the last idx of blocks
    transitions = np.concatenate([[0],
                                  np.where(delta_indices > 0)[0]+1,
                                  [len(indptr_idx)]])
    for ii in range(1, len(transitions), 1):
        indices_value = indices_chunk[transitions[ii-1]]
        if indices_value not in output_dict:
            output_dict[indices_value] = {'indices': [], 'data': []}
        indptr_values = indptr_idx[transitions[ii-1]: transitions[ii]]
        output_dict[indices_value]['indices'].append(indptr_values)
        if data_chunk is not None:
            data_values = data_chunk[transitions[ii-1]:transitions[ii]]
            output_dict[indices_value]['data'].append(data_values)
