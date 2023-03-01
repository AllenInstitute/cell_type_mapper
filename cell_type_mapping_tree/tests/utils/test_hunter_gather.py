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


from hierarchical_mapping.utils.h5ad_remapper import (
    rearrange_sparse_h5ad_hunter_gather,
    _merge_bounds)


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
def row_order_fixture(sparse_data_fixture):
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

    row_order = []
    for chunk in row_chunk_list:
        row_order += chunk
    return row_order


@pytest.mark.parametrize(
    "zero_out, output_chunks, n_row_collectors, buffer_size, "
    "read_in_size, verbose",
    ((False, 17, 3, 2000, 1000, False),
     (True, 17, 3, 2000, 1000, False),
     (False, 45, 1, 1000, 10000, False),
     (True, 45, 1, 1000, 40000, True)))
def test_rearrange_sparse_h5ad(
        zero_out,
        output_chunks,
        n_row_collectors,
        buffer_size,
        read_in_size,
        verbose,
        sparse_data_fixture,
        row_order_fixture):

    tmp_input_path = tempfile.mkstemp(prefix='input_', suffix='.h5ad')
    os.close(tmp_input_path[0])
    tmp_input_path = pathlib.Path(tmp_input_path[1])

    tmp_output_dir = tempfile.mkdtemp(prefix='output_', suffix='.zarr')

    data = np.copy(sparse_data_fixture)

    if zero_out:
        data[156,:] = 0.0

    base_csr = scipy_sparse.csr_matrix(data)
    a_data = anndata.AnnData(X=base_csr)
    a_data.write_h5ad(tmp_input_path)

    new_data = np.zeros(data.shape, dtype=data.dtype)
    for n_row, i_row in enumerate(row_order_fixture):
        new_data[n_row, :] = data[i_row, :]

    rearrange_sparse_h5ad_hunter_gather(
         h5ad_path=tmp_input_path,
         output_path=tmp_output_dir,
         row_order=row_order_fixture,
         chunks=output_chunks,
         n_row_collectors=n_row_collectors,
         buffer_size=buffer_size,
         read_in_size=read_in_size,
         verbose=verbose)

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
    assert new_dense.sum() > 0.0

    np.testing.assert_allclose(new_data, new_dense)

    tmp_input_path.unlink()
    _clean_up(tmp_output_dir)


def test_merge_bounds():
    in_bounds = [[(0, 1)], [(3, 4)], [(5, 6)]]
    out_bounds = [[(0, 1)], [(2, 3)], [(1, 7)]]
    new_in, new_out = _merge_bounds(in_bounds, out_bounds)
    assert len(new_in) == 2
    assert len(new_out) == 2
    assert [(0, 1), (1, 7)] in new_out
    assert [(2, 3)] in new_out
    assert [(0, 1), (5, 6)] in new_in
    assert [(3, 4)] in new_in
    return

    in_bounds = [[(0, 1)], [(3, 4)], [(5, 6)], [(18, 21)]]
    out_bounds = [[(0, 1)], [(2, 3)], [(1, 7)], [(7, 13)]]
    new_in, new_out = _merge_bounds(in_bounds, out_bounds)
    assert len(new_out) == 2
    assert len(new_in) == 2

    assert[(0, 1), (1, 7), (7, 13)] in new_out
    assert [(2, 3)] in new_out
    assert [(0, 1), (5, 6), (18, 21)] in new_in
    assert [(3, 4)] in new_in

    in_bounds = [[(0, 1)], [(3, 4)], [(5, 6)], [(18, 21)], [(99, 101)]]
    out_bounds = [[(0, 1)], [(2, 3)], [(1, 7)], [(7, 13)], [(2, 77)]]
    new_in, new_out = _merge_bounds(in_bounds, out_bounds)
    assert len(new_in) == 2
    assert len(new_out) == 2
    assert [(2, 3), (2, 77)] in new_out
    assert [(0, 1), (1, 7), (7, 13)] in new_out
    assert [(3, 4), (99, 101)] in new_in
    assert [(0, 1), (5, 6), (18, 21)] in new_in
