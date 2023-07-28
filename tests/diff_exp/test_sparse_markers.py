import pytest

import itertools
import numpy as np
import scipy.sparse as scipy_sparse

from cell_type_mapper.binary_array.binary_array import (
    BinarizedBooleanArray)

from cell_type_mapper.diff_exp.sparse_markers_by_pair import (
    can_we_make_sparse,
    sparse_markers_by_pair_from_arrays,
    SparseMarkersByPair)

from cell_type_mapper.diff_exp.sparse_markers_by_gene import (
    SparseMarkersByGene)


def test_can_we_make_sparse_no():

    rng = np.random.default_rng(88123)

    bad_gb = 0.0001
    n_rows =2**7
    n_cols = 1+np.ceil(bad_gb*8*1024**3/n_rows).astype(int)
    n_int = np.ceil(n_cols/8).astype(int)
    data = 255*np.ones((n_rows, n_int), dtype=np.uint8)
    marker_array = BinarizedBooleanArray.from_data_array(
        data_array=data,
        n_cols=n_cols)

    up_array = BinarizedBooleanArray.from_data_array(
        n_cols=n_cols,
        data_array=rng.integers(0, 255, (n_rows, n_int), dtype=np.uint8))

    actual = can_we_make_sparse(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=bad_gb)

    assert actual is None


@pytest.mark.parametrize(
    'expected_dtype', [np.uint8, np.uint16])
def test_can_we_make_sparse_yes(expected_dtype):

    rng = np.random.default_rng(22314)

    if expected_dtype == np.uint8:
        n_bits = 8
    else:
        n_bits = 16
    n_rows = 2**(n_bits-1)
    n_cols = 55

    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)

    marker_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    up_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    for i_row in range(n_rows):
        marker_array.set_row(i_row, marker_truth[i_row, :])
        up_array.set_row(i_row, up_truth[i_row, :])

    gb_good = 1.01*(16*marker_truth.sum()/(8*1024**3))

    actual = can_we_make_sparse(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=gb_good)

    assert actual is not None
    assert actual['gene_dtype'] == expected_dtype

    for i_col in range(n_cols):
        marker_col = marker_truth[:, i_col]
        up_col = up_truth[:, i_col]
        assert actual['up_col_sum'][i_col] == (marker_col*up_col).sum()
        assert actual['down_col_sum'][i_col] == (marker_col*(~up_col)).sum()


def test_make_sparse_by_pairs():
    n_bits = 8
    rng = np.random.default_rng(118231)
    n_rows = 2**(n_bits-1)
    n_cols = 112
    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    up_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    marker_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    for i_row in range(n_rows):
        up_array.set_row(i_row, up_truth[i_row, :])
        marker_array.set_row(i_row, marker_truth[i_row, :])

    gb_estimate = 1.001*(n_bits*marker_truth.sum()/(8*1024**3))

    sparse = sparse_markers_by_pair_from_arrays(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=gb_estimate)

    assert sparse is not None

    up_v = sparse['up_values']
    up_idx = sparse['up_idx']
    down_v = sparse['down_values']
    down_idx = sparse['down_idx']

    for i_col in range(n_cols):
        marker_col = marker_truth[:, i_col]
        up_col = up_truth[:, i_col]

        up_values = np.where(
            np.logical_and(marker_col, up_col))[0]
        i0 = up_idx[i_col]
        if i_col < (n_cols+1):
            i1 = up_idx[i_col+1]
        else:
            i1 = n_cols
        actual = sparse['up_values'][i0:i1]
        np.testing.assert_array_equal(actual, up_values)

        down_values = np.where(
            np.logical_and(marker_col, ~up_col))[0]
        i0 = down_idx[i_col]
        if i_col < (n_cols+1):
            i1 = down_idx[i_col+1]
        else:
            i1 = n_cols
        actual = sparse['down_values'][i0:i1]
        np.testing.assert_array_equal(actual, down_values)


def test_sparse_by_pairs_class():
    n_bits = 8
    expected_dtype = np.uint8
    rng = np.random.default_rng(118231)
    n_rows = 2**(n_bits-1)
    n_cols = 112
    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    up_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    marker_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    for i_row in range(n_rows):
        up_array.set_row(i_row, up_truth[i_row, :])
        marker_array.set_row(i_row, marker_truth[i_row, :])

    sparse = sparse_markers_by_pair_from_arrays(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=22)

    marker_sparse = SparseMarkersByPair(
        gene_idx=sparse['up_values'],
        pair_idx=sparse['up_idx'])

    ct = 0
    for i_col in range(n_cols):
        actual = marker_sparse.get_genes_for_pair(i_col)
        expected = np.where(
            np.logical_and(
                marker_truth[:, i_col],
                up_truth[:, i_col]))[0]
        ct += len(actual)
        np.testing.assert_array_equal(
            actual, expected)
        assert actual.dtype == expected_dtype
    assert ct > 10


@pytest.mark.parametrize(
    "pairs_to_keep,in_place",
    itertools.product(
        [np.array([0, 7, 17, 18, 23, 32, 45]),
         np.array([5, 16, 17, 77, 89]),
         np.array([11, 17, 45, 66, 111]),
         np.array([0, 17, 44, 53, 111])],
        [True, False]
    ))
def test_sparse_by_pairs_class_downsample_pairs(
        pairs_to_keep,
        in_place):
    n_bits = 8
    expected_dtype = np.uint8
    rng = np.random.default_rng(118231)
    rng.shuffle(pairs_to_keep)
    n_rows = 2**(n_bits-1)
    n_cols = 112
    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    up_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    marker_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    for i_row in range(n_rows):
        up_array.set_row(i_row, up_truth[i_row, :])
        marker_array.set_row(i_row, marker_truth[i_row, :])

    sparse = sparse_markers_by_pair_from_arrays(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=22)

    marker_sparse = SparseMarkersByPair(
        gene_idx=sparse['up_values'],
        pair_idx=sparse['up_idx'])

    output = marker_sparse.keep_only_pairs(
        pairs_to_keep,
        in_place=in_place)

    if in_place:
        assert output is None
        assert not np.array_equal(
            marker_sparse.gene_idx,
            sparse['up_values'])
        assert not np.array_equal(
            marker_sparse.pair_idx,
            sparse['up_idx'])
    else:
        assert not np.array_equal(
            output.gene_idx,
            sparse['up_values'])
        assert not np.array_equal(
            output.pair_idx,
            sparse['up_idx'])

        np.testing.assert_array_equal(
            marker_sparse.gene_idx,
            sparse['up_values'])
        np.testing.assert_array_equal(
            marker_sparse.pair_idx,
            sparse['up_idx'])

        marker_sparse = output

    ct = 0
    for i_new, i_old in enumerate(pairs_to_keep):
        actual = marker_sparse.get_genes_for_pair(i_new)
        expected = np.where(
            np.logical_and(
                marker_truth[:, i_old],
                up_truth[:, i_old]))[0]
        ct += len(actual)
        np.testing.assert_array_equal(
            actual, expected)
        assert actual.dtype == expected_dtype
    assert ct > 10


@pytest.mark.parametrize(
    "genes_to_keep, in_place",
    itertools.product(
        [np.array([0, 7, 17, 18, 23, 32, 45, 113], dtype=np.int64),
         np.array([5, 16, 17, 77, 89, 122], dtype=np.int64),
         np.array([11, 17, 45, 66, 111, 127], dtype=np.int64),
         np.array([0, 17, 44, 53, 111, 127], dtype=np.int64)],
        [True, False]))
def test_sparse_by_pairs_class_downsample_genes(
        genes_to_keep,
        in_place):
    n_bits = 8
    expected_dtype = np.uint8
    rng = np.random.default_rng(118231)
    rng.shuffle(genes_to_keep)
    n_rows = 2**(n_bits-1)
    n_cols = 112
    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    up_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    marker_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    for i_row in range(n_rows):
        up_array.set_row(i_row, up_truth[i_row, :])
        marker_array.set_row(i_row, marker_truth[i_row, :])

    sparse = sparse_markers_by_pair_from_arrays(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=22)

    marker_sparse = SparseMarkersByPair(
        gene_idx=sparse['up_values'],
        pair_idx=sparse['up_idx'])

    output = marker_sparse.keep_only_genes(
                genes_to_keep,
                in_place=in_place)

    if in_place:
        assert output is None
        assert not np.array_equal(
            marker_sparse.gene_idx,
            sparse['up_values'])
        assert not np.array_equal(
            marker_sparse.pair_idx,
            sparse['up_idx'])
    else:
        assert not np.array_equal(
            output.gene_idx,
            sparse['up_values'])
        assert not np.array_equal(
            output.pair_idx,
            sparse['up_idx'])

        np.testing.assert_array_equal(
            marker_sparse.gene_idx,
            sparse['up_values'])
        np.testing.assert_array_equal(
            marker_sparse.pair_idx,
            sparse['up_idx'])

        marker_sparse = output

    ct = 0
    for i_col in range(n_cols):
        actual = marker_sparse.get_genes_for_pair(i_col)
        expected = np.where(
            np.logical_and(
                marker_truth[:, i_col][genes_to_keep],
                up_truth[:, i_col][genes_to_keep]))[0]
        ct += len(actual)

        np.testing.assert_array_equal(
            actual, expected)
        assert actual.dtype == expected_dtype
    assert ct > 10


@pytest.mark.parametrize(
    "genes_to_keep, in_place",
    itertools.product(
        [np.array([0, 7, 17, 18, 23, 32, 45, 113], dtype=np.int64),
         np.array([5, 16, 17, 77, 89, 122], dtype=np.int64),
         np.array([11, 17, 45, 66, 111, 127], dtype=np.int64),
         np.array([0, 17, 44, 53, 111, 127], dtype=np.int64)],
        [True, False]))
def test_sparse_by_genes_class_downsample_genes(
        genes_to_keep,
        in_place):
    n_bits = 8
    expected_dtype = np.uint8
    rng = np.random.default_rng(118231)
    rng.shuffle(genes_to_keep)
    n_rows = 2**(n_bits-1)
    n_cols = 112
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    csr = scipy_sparse.csr_matrix(marker_truth)

    marker_sparse = SparseMarkersByGene(
        gene_idx=csr.indptr,
        pair_idx=csr.indices)

    output = marker_sparse.keep_only_genes(
                genes_to_keep,
                in_place=in_place)

    if in_place:
        assert output is None
        assert not np.array_equal(
            marker_sparse.gene_idx,
            csr.indptr)
        assert not np.array_equal(
            marker_sparse.pair_idx,
            csr.indices)
    else:
        assert not np.array_equal(
            output.gene_idx,
            csr.indptr)
        assert not np.array_equal(
            output.pair_idx,
            csr.indices)

        np.testing.assert_array_equal(
            marker_sparse.gene_idx,
            csr.indptr)
        np.testing.assert_array_equal(
            marker_sparse.pair_idx,
            csr.indices)

        marker_sparse = output

    ct = 0
    for ii, i_gene in enumerate(genes_to_keep):
        actual = marker_sparse.get_pairs_for_gene(ii)
        expected = np.where(marker_truth[i_gene, :])[0]
        ct += len(actual)

        np.testing.assert_array_equal(
            actual, expected)
    assert ct > 10


@pytest.mark.parametrize(
    "pairs_to_keep,in_place",
    itertools.product(
        [np.array([0, 7, 17, 18, 23, 32, 45]),
         np.array([5, 16, 17, 77, 89]),
         np.array([11, 17, 45, 66, 111]),
         np.array([0, 17, 44, 53, 111])],
        [True, False]
    ))
def test_sparse_by_genes_class_downsample_pairs(
        pairs_to_keep,
        in_place):
    n_bits = 8
    expected_dtype = np.uint8
    rng = np.random.default_rng(118231)
    rng.shuffle(pairs_to_keep)
    n_rows = 2**(n_bits-1)
    n_cols = 112
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    csr = scipy_sparse.csr_matrix(marker_truth)

    marker_sparse = SparseMarkersByGene(
        gene_idx=csr.indptr,
        pair_idx=csr.indices)

    output = marker_sparse.keep_only_pairs(
        pairs_to_keep,
        in_place=in_place)

    if in_place:
        assert output is None
        assert not np.array_equal(
            marker_sparse.gene_idx,
            csr.indptr)
        assert not np.array_equal(
            marker_sparse.pair_idx,
            csr.indices)
    else:
        assert not np.array_equal(
            output.gene_idx,
            csr.indptr)
        assert not np.array_equal(
            output.pair_idx,
            csr.indices)

        np.testing.assert_array_equal(
            marker_sparse.gene_idx,
            csr.indptr)
        np.testing.assert_array_equal(
            marker_sparse.pair_idx,
            csr.indices)

        marker_sparse = output

    ct = 0
    for i_gene in range(n_rows):
        actual = marker_sparse.get_pairs_for_gene(i_gene)
        expected = np.where(marker_truth[i_gene, :][pairs_to_keep])[0]
        ct += len(actual)
        np.testing.assert_array_equal(
            actual, expected)
    assert ct > 10
