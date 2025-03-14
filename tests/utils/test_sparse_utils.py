import pytest
import numpy as np
import scipy.sparse as scipy_sparse
import anndata
import h5py
import warnings

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.utils.sparse_utils import (
    load_csr,
    load_csc,
    load_csr_chunk,
    merge_csr,
    _load_disjoint_csr,
    precompute_indptr,
    downsample_indptr,
    mask_indptr_by_indices)


def test_load_csr(tmp_dir_fixture):
    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    rng = np.random.default_rng(88123)

    data = np.zeros(60000, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((200, 300))

    csr = scipy_sparse.csr_matrix(data)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        ann = anndata.AnnData(csr, dtype=int)

    ann.write_h5ad(tmp_path)

    with h5py.File(tmp_path, 'r') as src:
        for r0 in range(0, 150, 47):
            r1 = min(200, r0+47)
            subset = load_csr(
                        row_spec=(r0, r1),
                        n_cols=data.shape[1],
                        data=src['X/data'],
                        indices=src['X/indices'],
                        indptr=src['X/indptr'])
            np.testing.assert_array_equal(
                subset,
                data[r0:r1, :])

        for ii in range(10):
            r0 = rng.integers(3, 50)
            r1 = min(data.shape[0], r0+rng.integers(17, 81))

            subset = load_csr(
                row_spec=(r0, r1),
                n_cols=data.shape[1],
                data=src['X/data'],
                indices=src['X/indices'],
                indptr=src['X/indptr'])

            np.testing.assert_array_equal(
                subset,
                data[r0:r1, :])

    _clean_up(tmp_path)


def test_load_csc(tmp_dir_fixture):
    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5ad')

    rng = np.random.default_rng(5656213)

    data = np.zeros(60000, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((200, 300))

    csc = scipy_sparse.csc_matrix(data)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        ann = anndata.AnnData(csc, dtype=int)

    ann.write_h5ad(tmp_path)

    with h5py.File(tmp_path, 'r') as src:
        for c0 in range(0, 250, 47):
            c1 = min(300, c0+47)
            subset = load_csc(
                        col_spec=(c0, c1),
                        n_rows=data.shape[0],
                        data=src['X/data'],
                        indices=src['X/indices'],
                        indptr=src['X/indptr'])
            np.testing.assert_array_equal(
                subset,
                data[:, c0:c1])

        for ii in range(10):
            c0 = rng.integers(3, 50)
            c1 = min(data.shape[1], c0+rng.integers(17, 81))

            subset = load_csc(
                col_spec=(c0, c1),
                n_rows=data.shape[0],
                data=src['X/data'],
                indices=src['X/indices'],
                indptr=src['X/indptr'])

            np.testing.assert_array_equal(
                subset,
                data[:, c0:c1])

    _clean_up(tmp_path)


def test_load_csr_chunk(tmp_dir_fixture):
    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5ad')

    rng = np.random.default_rng(88123)

    data = np.zeros(60000, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((200, 300))

    csr = scipy_sparse.csr_matrix(data)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        ann = anndata.AnnData(csr, dtype=int)

    ann.write_h5ad(tmp_path)

    with h5py.File(tmp_path, 'r') as src:
        for r0 in range(0, 150, 47):
            r1 = min(200, r0+47)
            for c0 in range(0, 270, 37):
                c1 = min(300, c0+37)
                subset = load_csr_chunk(
                            row_spec=(r0, r1),
                            col_spec=(c0, c1),
                            data=src['X/data'],
                            indices=src['X/indices'],
                            indptr=src['X/indptr'])

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
                data=src['X/data'],
                indices=src['X/indices'],
                indptr=src['X/indptr'])

            assert subset.shape == (r1-r0, c1-c0)

            np.testing.assert_array_equal(
                subset,
                data[r0:r1, c0:c1])

    _clean_up(tmp_path)


def test_load_csr_chunk_very_sparse(tmp_dir_fixture):
    tmp_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5ad')

    data = np.zeros((20, 20), dtype=int)
    data[7, 11] = 1

    csr = scipy_sparse.csr_matrix(data)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        ann = anndata.AnnData(csr, dtype=int)

    ann.write_h5ad(tmp_path)

    with h5py.File(tmp_path, 'r') as src:
        for subset, expected_sum in zip([((1, 15), (6, 13)),
                                         ((15, 19), (3, 14))],
                                        [1, 0]):

            expected = data[subset[0][0]:subset[0][1],
                            subset[1][0]:subset[1][1]]

            actual = load_csr_chunk(
                    row_spec=(subset[0][0], subset[0][1]),
                    col_spec=(subset[1][0], subset[1][1]),
                    data=src['X/data'],
                    indices=src['X/indices'],
                    indptr=src['X/indptr'])

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


def test_load_disjoint_csr(tmp_dir_fixture):
    nrows = 200
    ncols = 300

    tmp_path = mkstemp_clean(
        dir=tmp_dir_fixture, suffix='.h5ad')

    rng = np.random.default_rng(776623)

    data = np.zeros(nrows*ncols, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((nrows, ncols))

    csr = scipy_sparse.csr_matrix(data)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        ann = anndata.AnnData(csr, dtype=int)

    ann.write_h5ad(tmp_path)

    index_list = np.unique(rng.integers(0, nrows, 45))

    rng.shuffle(index_list)
    expected = np.zeros((len(index_list), ncols), dtype=int)

    for ct, ii in enumerate(index_list):
        expected[ct, :] = data[ii, :]

    with h5py.File(tmp_path, 'r') as src:
        (chunk_data,
         chunk_indices,
         chunk_indptr) = _load_disjoint_csr(
                             row_index_list=index_list,
                             data=src['X/data'],
                             indices=src['X/indices'],
                             indptr=src['X/indptr'])

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
    "to_keep",
    [np.array([0, 19, 33, 11, 29, 443]),
     np.array([1, 7, 229, 8]),
     np.array([0, 66, 44, 549]),
     np.array([33, 14, 17, 549])])
def test_downsample_indptr(to_keep):
    rng = np.random.default_rng(887123)
    n_rows = 550
    n_cols = 127
    data = np.zeros(n_rows*n_cols, dtype=float)
    chosen = rng.choice(
        np.arange(n_rows*n_cols, dtype=int),
        n_rows*n_cols//3,
        replace=False)
    data[chosen] = rng.random(len(chosen))
    data = data.reshape(n_rows, n_cols)
    data = scipy_sparse.csr_matrix(data)

    (new_indptr,
     new_indices) = downsample_indptr(
         indptr_old=data.indptr,
         indices_old=data.indices,
         indptr_to_keep=to_keep)

    for ii, i_row in enumerate(to_keep):
        actual0 = new_indptr[ii]
        actual1 = new_indptr[ii+1]
        actual = new_indices[actual0:actual1]
        expected0 = data.indptr[i_row]
        expected1 = data.indptr[i_row+1]
        expected = data.indices[expected0:expected1]
        np.testing.assert_array_equal(actual, expected)


def test_mask_indptr():
    rng = np.random.default_rng(445713)
    n_rows = 550
    n_cols = 127
    data = np.zeros(n_rows*n_cols, dtype=float)
    chosen = rng.choice(
        np.arange(n_rows*n_cols, dtype=int),
        n_rows*n_cols//3,
        replace=False)
    data[chosen] = rng.random(len(chosen))
    data = data.reshape(n_rows, n_cols)
    data = scipy_sparse.csr_matrix(data)

    indices_map = {
        45: 0,
        117: 1,
        33: 2,
        95: 4,
        111: 5,
        23: 6,
        48: 11,
        39: 15
    }

    (new_indptr,
     new_indices) = mask_indptr_by_indices(
            indptr_old=data.indptr,
            indices_old=data.indices,
            indices_map=indices_map)

    assert len(new_indices) > 0
    for ii in range(len(data.indptr)-1):
        src0 = data.indptr[ii]
        src1 = data.indptr[ii+1]
        expected = [indices_map[n]
                    for n in data.indices[src0:src1]
                    if n in indices_map]
        expected = np.sort(expected)
        dst0 = new_indptr[ii]
        dst1 = new_indptr[ii+1]
        np.testing.assert_array_equal(
            expected,
            new_indices[dst0:dst1])
