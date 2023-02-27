import pytest
import numpy as np
import scipy.sparse as scipy_sparse
import anndata
import os
import h5py
import tempfile
import pathlib
import zarr

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.sparse_utils import(
    load_csr,
    load_csr_chunk,
    merge_csr,
    _load_disjoint_csr,
    precompute_indptr,
    remap_csr_matrix)


def test_load_csr():
    tmp_path = tempfile.mkdtemp(suffix='.zarr')

    rng = np.random.default_rng(88123)

    data = np.zeros(60000, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((200, 300))

    csr = scipy_sparse.csr_matrix(data)
    ann = anndata.AnnData(csr, dtype=int)
    ann.write_zarr(tmp_path)

    with zarr.open(tmp_path, 'r') as written_zarr:
        for r0 in range(0, 150, 47):
            r1 = min(200, r0+47)
            subset = load_csr(
                        row_spec=(r0, r1),
                        n_cols=data.shape[1],
                        data=written_zarr.X.data,
                        indices=written_zarr.X.indices,
                        indptr=written_zarr.X.indptr)
            np.testing.assert_array_equal(
                subset,
                data[r0:r1, :])

        for ii in range(10):
            r0 = rng.integers(3, 50)
            r1 = min(data.shape[0], r0+rng.integers(17, 81))

            subset = load_csr(
                row_spec=(r0, r1),
                n_cols=data.shape[1],
                data=written_zarr.X.data,
                indices=written_zarr.X.indices,
                indptr=written_zarr.X.indptr)

            np.testing.assert_array_equal(
                subset,
                data[r0:r1, :])

    _clean_up(tmp_path)


def test_load_csr_chunk():
    tmp_path = tempfile.mkdtemp(suffix='.zarr')

    rng = np.random.default_rng(88123)

    data = np.zeros(60000, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((200, 300))

    csr = scipy_sparse.csr_matrix(data)
    ann = anndata.AnnData(csr, dtype=int)
    ann.write_zarr(tmp_path)

    with zarr.open(tmp_path, 'r') as written_zarr:
        for r0 in range(0, 150, 47):
            r1 = min(200, r0+47)
            for c0 in range(0, 270, 37):
                c1 = min(300, c0+37)
                subset = load_csr_chunk(
                            row_spec=(r0, r1),
                            col_spec=(c0, c1),
                            data=written_zarr.X.data,
                            indices=written_zarr.X.indices,
                            indptr=written_zarr.X.indptr)

                assert subset.shape == (r1-r0, c1-c0)

                np.testing.assert_array_equal(
                    subset,
                    data[r0:r1, c0:c1])

        for ii in range(10):
            r0 = rng.integers(3, 50)
            r1 = min(data.shape[0], r0+rng.integers(17, 81))
            c0 = rng.integers(70, 220)
            c1 = min(data.shape[1], c0+rng.integers(20, 91))

            subset = load_csr_chunk(
                row_spec=(r0, r1),
                col_spec=(c0, c1),
                data=written_zarr.X.data,
                indices=written_zarr.X.indices,
                indptr=written_zarr.X.indptr)

            assert subset.shape == (r1-r0, c1-c0)

            np.testing.assert_array_equal(
                subset,
                data[r0:r1, c0:c1])

    _clean_up(tmp_path)


def test_load_csr_chunk_very_sparse():
    tmp_path = tempfile.mkdtemp(suffix='.zarr')

    data = np.zeros((20, 20), dtype=int)
    data[7, 11] = 1

    csr = scipy_sparse.csr_matrix(data)
    ann = anndata.AnnData(csr, dtype=int)
    ann.write_zarr(tmp_path)

    with zarr.open(tmp_path, 'r') as written_zarr:
        for subset, expected_sum in zip([((1, 15), (6, 13)),
                                         ((15, 19), (3, 14))],
                                        [1, 0]):

            expected = data[subset[0][0]:subset[0][1],
                            subset[1][0]:subset[1][1]]

            actual = load_csr_chunk(
                    row_spec=(subset[0][0], subset[0][1]),
                    col_spec=(subset[1][0], subset[1][1]),
                    data=written_zarr.X.data,
                    indices=written_zarr.X.indices,
                    indptr=written_zarr.X.indptr)

            assert actual.sum() == expected_sum

            np.testing.assert_array_equal(expected, actual)



def test_merge_csr():

    nrows = 100
    ncols = 234

    rng = np.random.default_rng(6123512)
    data = np.zeros((nrows*ncols), dtype=float)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//3,
                            replace=False)
    data[chosen_dex] = rng.random(len(chosen_dex))
    data = data.reshape((nrows, ncols))

    final_csr = scipy_sparse.csr_matrix(data)

    sub0 = scipy_sparse.csr_matrix(data[:32, :])
    sub1 = scipy_sparse.csr_matrix(data[32:71, :])
    sub2 = scipy_sparse.csr_matrix(data[71:, :])

    (merged_data,
     merged_indices,
     merged_indptr) = merge_csr(
         data_list=[sub0.data, sub1.data, sub2.data],
         indices_list=[sub0.indices, sub1.indices, sub2.indices],
         indptr_list=[sub0.indptr, sub1.indptr, sub2.indptr])


    np.testing.assert_allclose(merged_data, final_csr.data)
    np.testing.assert_array_equal(merged_indices, final_csr.indices)
    np.testing.assert_array_equal(merged_indptr, final_csr.indptr)


    merged_csr = scipy_sparse.csr_matrix(
        (merged_data, merged_indices, merged_indptr),
        shape=(nrows, ncols))

    result = merged_csr.todense()
    np.testing.assert_allclose(result, data)


@pytest.mark.parametrize("zero_block", (0, 1, 2))
def test_merge_csr_block_zeros(zero_block):

    nrows = 100
    ncols = 234

    rng = np.random.default_rng(6123512)
    data = np.zeros((nrows*ncols), dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//3,
                            replace=False)
    data[chosen_dex] = rng.integers(3, 6000000, len(chosen_dex))
    data = data.reshape((nrows, ncols))

    if zero_block == 0:
        data[:32, :] = 0
    elif zero_block == 1:
        data[32:71, :] = 0
    elif zero_block == 2:
        data[71:, :] = 0

    final_csr = scipy_sparse.csr_matrix(data)

    sub0 = scipy_sparse.csr_matrix(data[:32, :])
    sub1 = scipy_sparse.csr_matrix(data[32:71, :])
    sub2 = scipy_sparse.csr_matrix(data[71:, :])

    (merged_data,
     merged_indices,
     merged_indptr) = merge_csr(
         data_list=[sub0.data, sub1.data, sub2.data],
         indices_list=[sub0.indices, sub1.indices, sub2.indices],
         indptr_list=[sub0.indptr, sub1.indptr, sub2.indptr])


    np.testing.assert_allclose(merged_data, final_csr.data)
    np.testing.assert_array_equal(merged_indices, final_csr.indices)
    np.testing.assert_array_equal(merged_indptr, final_csr.indptr)


    merged_csr = scipy_sparse.csr_matrix(
        (merged_data, merged_indices, merged_indptr),
        shape=(nrows, ncols))

    result = merged_csr.todense()
    np.testing.assert_allclose(result, data)



def test_load_disjoint_csr():
    nrows = 200
    ncols = 300

    tmp_path = tempfile.mkdtemp(suffix='.zarr')

    rng = np.random.default_rng(776623)

    data = np.zeros(nrows*ncols, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((nrows, ncols))

    csr = scipy_sparse.csr_matrix(data)
    ann = anndata.AnnData(csr, dtype=int)
    ann.write_zarr(tmp_path)

    index_list = np.unique(rng.integers(0, nrows, 45))

    rng.shuffle(index_list)
    expected = np.zeros((len(index_list), ncols), dtype=int)

    for ct, ii in enumerate(index_list):
        expected[ct, :] = data[ii, :]

    with zarr.open(tmp_path, 'r') as written_zarr:
        (chunk_data,
         chunk_indices,
         chunk_indptr) = _load_disjoint_csr(
                             row_index_list=index_list,
                             data=written_zarr.X.data,
                             indices=written_zarr.X.indices,
                             indptr=written_zarr.X.indptr)

    actual = scipy_sparse.csr_matrix(
                (chunk_data, chunk_indices, chunk_indptr),
                shape=(len(index_list), ncols)).todense()

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected)

    _clean_up(tmp_path)


def test_precompute_indptr():
    rng = np.random.default_rng(87123331)
    nrows = 112
    ncols = 235
    data = np.zeros((nrows*ncols), dtype=float)
    chosen_dex = rng.choice(np.arange(nrows*ncols),
                            nrows*ncols//7,
                            replace=False)
    data[chosen_dex] = rng.random(len(chosen_dex))
    data = data.reshape(nrows, ncols)
    baseline_csr = scipy_sparse.csr_array(data)

    row_reorder = np.arange(nrows, dtype=int)
    rng.shuffle(row_reorder)

    new_data = np.zeros(data.shape, dtype=float)
    for ii, rr in enumerate(row_reorder):
        new_data[ii, :] = data[rr, :]
    assert not np.allclose(new_data, data)
    new_csr = scipy_sparse.csr_array(new_data)

    actual = precompute_indptr(
                indptr_in=baseline_csr.indptr,
                row_order=row_reorder)

    np.testing.assert_array_equal(actual, new_csr.indptr)


@pytest.mark.parametrize(
    "flush_every", (200, 23, 150))
def test_remap_csr_matrix(
        tmp_path_factory,
        flush_every):

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('remap_csr'))

    rng = np.random.default_rng(77662233)
    nrows = 112
    ncols = 235
    data = np.zeros((nrows*ncols), dtype=float)
    chosen_dex = rng.choice(np.arange(nrows*ncols),
                            nrows*ncols//7,
                            replace=False)
    data[chosen_dex] = rng.random(len(chosen_dex))
    data = data.reshape(nrows, ncols)
    baseline_csr = scipy_sparse.csr_array(data)

    baseline_h5_path = tmp_dir / 'baseline.h5'
    with h5py.File(baseline_h5_path, 'w') as out_file:
        out_file.create_dataset('data', data=baseline_csr.data)
        out_file.create_dataset('indices', data=baseline_csr.indices)

    row_reorder = np.arange(nrows, dtype=int)
    rng.shuffle(row_reorder)

    new_dense_data = np.zeros(data.shape, dtype=float)
    for ii, rr in enumerate(row_reorder):
        new_dense_data[ii, :] = data[rr, :]
    assert not np.allclose(new_dense_data, data)
    new_csr = scipy_sparse.csr_array(new_dense_data)

    new_data = np.zeros(baseline_csr.data.shape,
                        dtype=baseline_csr.data.dtype)
    new_indices = np.zeros(baseline_csr.indices.shape,
                           dtype=int)
    new_indptr = np.zeros(baseline_csr.indptr.shape,
                          dtype=int)

    reordered_indptr = precompute_indptr(
                            indptr_in=baseline_csr.indptr,
                            row_order=row_reorder)

    with h5py.File(baseline_h5_path, 'r') as in_file:
        remap_csr_matrix(
            data_handle=in_file['data'],
            indices_handle=in_file['indices'],
            indptr=baseline_csr.indptr,
            new_indptr=reordered_indptr,
            new_row_order=row_reorder,
            data_output_handle=new_data,
            indices_output_handle=new_indices,
            indptr_output_handle=new_indptr,
            flush_every=flush_every)

    np.testing.assert_allclose(new_data, new_csr.data)
    np.testing.assert_array_equal(new_indices, new_csr.indices)
    np.testing.assert_array_equal(new_indptr, new_csr.indptr)

    _clean_up(tmp_dir)
