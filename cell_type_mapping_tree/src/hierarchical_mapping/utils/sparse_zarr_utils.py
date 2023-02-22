import zarr

from hierarchical_mapping.utils.sparse_utils import (
    _merge_csr_chunk,
    _load_disjoint_csr)


def rearrange_sparse_zarr(
        input_path,
        output_path,
        row_chunk_list,
        chunks=5000):

    with zarr.open(input_path, 'r') as input_zarr:
        data_shape = input_zarr['data'].shape
        indptr_shape = input_zarr['indptr'].shape
        with zarr.open(output_path, 'w') as output_zarr:
            output_zarr.create(
                        name='data',
                        shape=data_shape,
                        dtype=input_zarr['data'].dtype,
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

            _rearrange_sparse_zarr(
                data_in = input_zarr['data'],
                indices_in = input_zarr['indices'],
                indptr_in = input_zarr['indptr'],
                row_chunk_list = row_chunk_list,
                data_out = output_zarr['data'],
                indices_out = output_zarr['indices'],
                indptr_out = output_zarr['indptr'])

            output_zarr['indptr'][-1] = data_shape[0]


def _rearrange_sparse_zarr(
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
