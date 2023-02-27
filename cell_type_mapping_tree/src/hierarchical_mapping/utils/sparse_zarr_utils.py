import zarr
import h5py

from hierarchical_mapping.utils.sparse_utils import (
    _merge_csr_chunk,
    _load_disjoint_csr)


def rearrange_sparse_zarr(
        input_path,
        output_path,
        row_chunk_list,
        chunks=5000):

    with zarr.open(input_path, 'r') as input_zarr:
        write_rearranged_zarr(
            data_handle=input_zarr['data'],
            indices_handle=input_zarr['indices'],
            indptr_handle=input_zarr['indptr'],
            output_path=output_path,
            row_chunk_list=row_chunk_list,
            chunks=chunks)

def rearrange_sparse_h5ad(
        h5ad_path,
        output_path,
        row_chunk_list,
        chunks=5000):

    with h5py.File(h5ad_path, 'r', swmr=True) as input_handle:
        write_rearranged_zarr(
            data_handle=input_handle['X']['data'],
            indices_handle=input_handle['X']['indices'],
            indptr_handle=input_handle['X']['indptr'],
            output_path=output_path,
            row_chunk_list=row_chunk_list,
            chunks=chunks)

def write_rearranged_zarr(
        data_handle,
        indices_handle,
        indptr_handle,
        output_path,
        row_chunk_list,
        chunks=5000):

    data_shape = data_handle.shape
    indptr_shape = indptr_handle.shape
    with zarr.open(output_path, 'w') as output_zarr:
        output_zarr.create(
                    name='data',
                    shape=data_shape,
                    dtype=data_handle.dtype,
                    chunks=chunks)
        output_zarr.create(
                    name='indices',
                    shape=data_shape,
                    dtype=int,
                    chunks=chunks)
        output_zarr.create(
                    name='indptr',
                    shape=indptr_shape,
                    dtype=int,
                    chunks=chunks)

        _rearrange_sparse_data(
            data_in = data_handle,
            indices_in = indices_handle,
            indptr_in = indptr_handle,
            row_chunk_list = row_chunk_list,
            data_out = output_zarr['data'],
            indices_out = output_zarr['indices'],
            indptr_out = output_zarr['indptr'])

        output_zarr['indptr'][-1] = data_shape[0]


def _rearrange_sparse_data(
        data_in,
        indices_in,
        indptr_in,
        row_chunk_list,
        data_out,
        indices_out,
        indptr_out):

    idx0 = 0
    ptr0 = 0

    for row_chunk in row_chunk_list:

        (this_data,
         this_indices,
         this_indptr) = _load_disjoint_csr(
                             row_index_list=row_chunk,
                             data=data_in,
                             indices=indices_in,
                             indptr=indptr_in)

        (data_out,
         indices_out,
         indptr_out,
         idx1,
         ptr1) = _merge_csr_chunk(
                    data_in=this_data,
                    indices_in=this_indices,
                    indptr_in=this_indptr,
                    data=data_out,
                    indices=indices_out,
                    indptr=indptr_out,
                    idx0=idx0,
                    ptr0=ptr0)

        idx0 = idx1
        ptr0 = ptr1
