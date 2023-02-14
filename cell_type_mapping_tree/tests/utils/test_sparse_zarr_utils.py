import pytest
import pathlib
import zarr
import numpy as np
import tempfile
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.sparse_zarr_utils import (
    rearrange_sparse_zarr)


def _clean_up(target_path):
    target_path = pathlib.Path(target_path)
    if target_path.is_file():
        target_path.unlink()
    elif target_path.is_dir():
        for sub_path in target_path.iterdir():
            _clean_up(sub_path)
        target_path.rmdir()


@pytest.mark.parametrize('zero_out', (True, False))
def test_rearrange_sparse_zarr(zero_out):
    tmp_input_dir = tempfile.mkdtemp(prefix='input_', suffix='.zarr')
    tmp_output_dir = tempfile.mkdtemp(prefix='output_', suffix='.zarr')

    rng = np.random.default_rng(772334)
    nrows = 314
    ncols = 567

    data = np.zeros((nrows*ncols), dtype=int)
    chosen_dex = rng.choice(np.arange(nrows*ncols),
                            nrows*ncols//7,
                            replace=False)
    data[chosen_dex] = rng.integers(0, 2000000, len(chosen_dex))
    data = data.reshape((nrows, ncols))

    row_indexes = np.arange(nrows)

    row_chunk_list = []
    for ii in range(5):
        row_chunk_list.append([])

    for idx in row_indexes:
        jj = rng.integers(0, len(row_chunk_list))
        row_chunk_list[jj].append(idx)

    for ii in range(len(row_chunk_list)):
        assert len(row_chunk_list[ii]) > 0

    if zero_out:
        for i_row in row_chunk_list[2]:
            data[i_row, :] = 0

    base_csr = scipy_sparse.csr_matrix(data)
    with zarr.open(tmp_input_dir, 'w') as input_zarr:
        input_zarr['data'] = zarr.create(base_csr.data.shape,
                                         dtype=base_csr.data.dtype)
        input_zarr['data'][:] = base_csr.data[:]
        input_zarr['indices'] = zarr.create(base_csr.indices.shape, dtype=int)
        input_zarr['indices'][:] = base_csr.indices[:]
        input_zarr['indptr'] = zarr.create(base_csr.indptr.shape, dtype=int)
        input_zarr['indptr'][:] = base_csr.indptr[:]

    new_data = np.zeros(data.shape, dtype=data.dtype)
    n_row = 0
    for chunk in row_chunk_list:
        for i_row in chunk:
            new_data[n_row, :] = data[i_row, :]
            n_row += 1

    rearrange_sparse_zarr(
         input_path=tmp_input_dir,
         output_path=tmp_output_dir,
         row_chunk_list=row_chunk_list,
         chunks=50)

    with zarr.open(tmp_output_dir, 'r') as output_zarr:
        data_r = np.array(output_zarr['data'])
        indices_r = np.array(output_zarr['indices'])
        indptr_r = np.array(output_zarr['indptr'])

    new_csr = scipy_sparse.csr_matrix(
                    (data_r, indices_r, indptr_r),
                    shape=(nrows, ncols))
    new_dense = new_csr.toarray()

    np.testing.assert_allclose(new_data, new_dense)

    _clean_up(tmp_input_dir)
    _clean_up(tmp_output_dir)
