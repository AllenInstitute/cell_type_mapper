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

from cell_type_mapper.utils.csc_to_csr import (
    transpose_sparse_matrix_on_disk
)


def transpose_sparse_matrix_on_disk_v2(
        h5_path,
        indices_tag,
        indptr_tag,
        data_tag,
        indices_max,
        max_gb,
        output_path,
        verbose=False,
        tmp_dir=None,
        n_processors=4):
    """
    indices_max is the number of unique indices values in original array
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
            indices_max=indices_max,
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
        indices_max,
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

    gb_per_process = 0.8 * max_gb // n_processors
    indices_chunk_size = np.ceil(indices_max / n_processors).astype(int)

    path_list = []
    process_list = []
    for i0 in range(0, indices_max, indices_chunk_size):

        i1 = min(indices_max, i0+indices_chunk_size)

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
                'indices_max': indices_max,
                'indices_slice': (i0, i1),
                'output_path': tmp_path,
                'max_gb': gb_per_process
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

        chunk_size = 10000
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
        indices_max,
        indices_slice,
        output_path,
        max_gb=12):
    use_data = (data_tag is not None)

    with h5py.File(h5_path, 'r', swmr=True) as src:
        indices_handle = src[indices_tag]
        indptr_handle = src[indptr_tag]
        if use_data:
            data_handle = src[data_tag]
        else:
            data_handle = None

        transpose_sparse_matrix_on_disk(
            indices_handle=indices_handle,
            indptr_handle=indptr_handle,
            data_handle=data_handle,
            indices_max=indices_max,
            max_gb=max_gb,
            output_path=output_path,
            verbose=False,
            indices_slice=indices_slice)

    with h5py.File(output_path, 'a') as dst:
        dst.create_dataset(
            'indices_slice',
            data=np.array(indices_slice))
