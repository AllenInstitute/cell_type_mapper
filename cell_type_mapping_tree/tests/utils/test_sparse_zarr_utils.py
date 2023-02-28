import pytest
import pathlib
import zarr
import os
import json
import numpy as np
import tempfile
import scipy.sparse as scipy_sparse
import anndata

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.sparse_zarr_utils import (
    rearrange_sparse_zarr,
    rearrange_sparse_h5ad)

@pytest.fixture(scope='session')
def sparse_data_fixture():

    rng = np.random.default_rng(772334)
    nrows = 314
    ncols = 567

    data = np.zeros((nrows*ncols), dtype=int)
    chosen_dex = rng.choice(np.arange(nrows*ncols),
                            nrows*ncols//7,
                            replace=False)
    data[chosen_dex] = rng.integers(0, 2000000, len(chosen_dex))
    data = data.reshape((nrows, ncols))
    return data


@pytest.fixture(scope='session')
def row_chunk_list_fixture(sparse_data_fixture):
    data = sparse_data_fixture
    rng = np.random.default_rng(881231)

    row_indexes = np.arange(data.shape[0])

    row_chunk_list = []
    for ii in range(5):
        row_chunk_list.append([])

    for idx in row_indexes:
        jj = rng.integers(0, len(row_chunk_list))
        row_chunk_list[jj].append(idx)

    for ii in range(len(row_chunk_list)):
        assert len(row_chunk_list[ii]) > 0

    return row_chunk_list

@pytest.mark.parametrize(
        'zero_out, output_chunks',
        [(True, 17), (True, 23), (False, 33), (False, 14)])
def test_rearrange_sparse_zarr(
        zero_out,
        output_chunks,
        sparse_data_fixture,
        row_chunk_list_fixture):
    tmp_input_dir = tempfile.mkdtemp(prefix='input_', suffix='.zarr')
    tmp_output_dir = tempfile.mkdtemp(prefix='output_', suffix='.zarr')

    data = np.copy(sparse_data_fixture)
    row_chunk_list = row_chunk_list_fixture

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
         chunks=output_chunks)

    # verify that chunks were properly set
    for pth in ('data', 'indices', 'indptr'):
        full_pth = pathlib.Path(tmp_output_dir) / pth
        config = json.load(open(full_pth/'.zarray', 'rb'))
        assert config['chunks'] == [output_chunks]

    with zarr.open(tmp_output_dir, 'r') as output_zarr:
        data_r = np.array(output_zarr['data'])
        indices_r = np.array(output_zarr['indices'])
        indptr_r = np.array(output_zarr['indptr'])

    new_csr = scipy_sparse.csr_matrix(
                    (data_r, indices_r, indptr_r),
                    shape=data.shape)
    new_dense = new_csr.toarray()

    np.testing.assert_allclose(new_data, new_dense)

    _clean_up(tmp_input_dir)
    _clean_up(tmp_output_dir)



@pytest.mark.parametrize(
        'zero_out, output_chunks, flush_every, n_processors',
        [(True, 17, 100, 1),
         (True, 23, 500, 1),
         (False, 33, 314, 1),
         (False, 14, 26, 1),
         (True, 47, 113, 3),
         (False, 47, 113, 3)])
def test_rearrange_sparse_h5ad(
        zero_out,
        output_chunks,
        flush_every,
        n_processors,
        sparse_data_fixture,
        row_chunk_list_fixture):

    tmp_input_path = tempfile.mkstemp(prefix='input_', suffix='.h5ad')
    os.close(tmp_input_path[0])
    tmp_input_path = pathlib.Path(tmp_input_path[1])

    tmp_output_dir = tempfile.mkdtemp(prefix='output_', suffix='.zarr')

    data = np.copy(sparse_data_fixture)
    row_chunk_list = row_chunk_list_fixture

    if zero_out:
        for i_row in row_chunk_list[2]:
            data[i_row, :] = 0

    base_csr = scipy_sparse.csr_matrix(data)
    a_data = anndata.AnnData(X=base_csr)
    a_data.write_h5ad(tmp_input_path)

    new_data = np.zeros(data.shape, dtype=data.dtype)
    n_row = 0
    for chunk in row_chunk_list:
        for i_row in chunk:
            new_data[n_row, :] = data[i_row, :]
            n_row += 1

    rearrange_sparse_h5ad(
         h5ad_path=tmp_input_path,
         output_path=tmp_output_dir,
         row_chunk_list=row_chunk_list,
         chunks=output_chunks,
         flush_every=flush_every,
         n_processors=n_processors)

    # verify that chunks were properly set
    for pth in ('data', 'indices', 'indptr'):
        full_pth = pathlib.Path(tmp_output_dir) / pth
        config = json.load(open(full_pth/'.zarray', 'rb'))
        assert config['chunks'] == [output_chunks]

    with zarr.open(tmp_output_dir, 'r') as output_zarr:
        data_r = np.array(output_zarr['data'])
        indices_r = np.array(output_zarr['indices'])
        indptr_r = np.array(output_zarr['indptr'])

    new_csr = scipy_sparse.csr_matrix(
                    (data_r, indices_r, indptr_r),
                    shape=data.shape)
    new_dense = new_csr.toarray()

    np.testing.assert_allclose(new_data, new_dense)

    tmp_input_path.unlink()
    _clean_up(tmp_output_dir)
