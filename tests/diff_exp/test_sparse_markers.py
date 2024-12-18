import pytest

import itertools
import numpy as np
import scipy.sparse as scipy_sparse

from cell_type_mapper.diff_exp.sparse_markers_by_pair import (
    SparseMarkersByPair)

from cell_type_mapper.diff_exp.sparse_markers_by_gene import (
    SparseMarkersByGene)


@pytest.fixture(scope='module')
def n_cols():
    return 122


@pytest.fixture(scope='module')
def n_rows():
    return 2**7


@pytest.fixture(scope='module')
def mask_array_fixture(n_cols, n_rows):
    rng = np.random.default_rng(118231)
    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    return np.logical_and(marker_truth, up_truth)


@pytest.fixture(scope='module')
def csc_mask_array_fixture(mask_array_fixture):
    return scipy_sparse.csc_array(mask_array_fixture)


@pytest.fixture(scope='module')
def csr_mask_array_fixture(mask_array_fixture):
    return scipy_sparse.csr_array(mask_array_fixture)


def test_sparse_by_pairs_class(
        mask_array_fixture,
        csc_mask_array_fixture,
        n_cols,
        n_rows):

    marker_sparse = SparseMarkersByPair(
        gene_idx=csc_mask_array_fixture.indices,
        pair_idx=csc_mask_array_fixture.indptr)

    ct = 0
    for i_col in range(n_cols):
        actual = marker_sparse.get_genes_for_pair(i_col)
        expected = np.where(
            mask_array_fixture[:, i_col])[0]
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
def test_sparse_by_pairs_class_downsample_pairs(
        pairs_to_keep,
        in_place,
        mask_array_fixture,
        csc_mask_array_fixture,
        n_rows,
        n_cols):

    baseline = SparseMarkersByPair(
        gene_idx=csc_mask_array_fixture.indices,
        pair_idx=csc_mask_array_fixture.indptr)

    marker_sparse = SparseMarkersByPair(
        gene_idx=csc_mask_array_fixture.indices,
        pair_idx=csc_mask_array_fixture.indptr)

    output = marker_sparse.keep_only_pairs(
        pairs_to_keep,
        in_place=in_place)

    if in_place:
        assert output is None
        assert not np.array_equal(
            marker_sparse.gene_idx,
            baseline.gene_idx)
        assert not np.array_equal(
            marker_sparse.pair_idx,
            baseline.pair_idx)
    else:
        assert not np.array_equal(
            output.gene_idx,
            baseline.gene_idx)
        assert not np.array_equal(
            output.pair_idx,
            baseline.pair_idx)

        np.testing.assert_array_equal(
            marker_sparse.gene_idx,
            baseline.gene_idx)
        np.testing.assert_array_equal(
            marker_sparse.pair_idx,
            baseline.pair_idx)

        marker_sparse = output

    ct = 0
    for i_new, i_old in enumerate(pairs_to_keep):
        actual = marker_sparse.get_genes_for_pair(i_new)
        expected = np.where(
            mask_array_fixture[:, i_old])[0]
        ct += len(actual)
        np.testing.assert_array_equal(
            actual, expected)
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
        in_place,
        mask_array_fixture,
        csc_mask_array_fixture,
        n_rows,
        n_cols):

    baseline = SparseMarkersByPair(
        gene_idx=csc_mask_array_fixture.indices,
        pair_idx=csc_mask_array_fixture.indptr)

    marker_sparse = SparseMarkersByPair(
        gene_idx=csc_mask_array_fixture.indices,
        pair_idx=csc_mask_array_fixture.indptr)

    output = marker_sparse.keep_only_genes(
                genes_to_keep,
                in_place=in_place)

    if in_place:
        assert output is None
        assert not np.array_equal(
            marker_sparse.gene_idx,
            baseline.gene_idx)
        assert not np.array_equal(
            marker_sparse.pair_idx,
            baseline.pair_idx)
    else:
        assert not np.array_equal(
            output.gene_idx,
            baseline.gene_idx)
        assert not np.array_equal(
            output.pair_idx,
            baseline.pair_idx)

        np.testing.assert_array_equal(
            marker_sparse.gene_idx,
            baseline.gene_idx)
        np.testing.assert_array_equal(
            marker_sparse.pair_idx,
            baseline.pair_idx)

        marker_sparse = output

    ct = 0
    for i_col in range(n_cols):
        actual = marker_sparse.get_genes_for_pair(i_col)
        expected = np.where(
            mask_array_fixture[:, i_col][genes_to_keep])[0]
        ct += len(actual)

        np.testing.assert_array_equal(
            actual, expected)
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
        in_place,
        mask_array_fixture,
        csr_mask_array_fixture,
        n_cols,
        n_rows):

    marker_sparse = SparseMarkersByGene(
        gene_idx=csr_mask_array_fixture.indptr,
        pair_idx=csr_mask_array_fixture.indices)

    output = marker_sparse.keep_only_pairs(
        pairs_to_keep,
        in_place=in_place)

    if in_place:
        assert output is None
        assert not np.array_equal(
            marker_sparse.gene_idx,
            csr_mask_array_fixture.indptr)
        assert not np.array_equal(
            marker_sparse.pair_idx,
            csr_mask_array_fixture.indices)
    else:
        assert not np.array_equal(
            output.gene_idx,
            csr_mask_array_fixture.indptr)
        assert not np.array_equal(
            output.pair_idx,
            csr_mask_array_fixture.indices)

        np.testing.assert_array_equal(
            marker_sparse.gene_idx,
            csr_mask_array_fixture.indptr)
        np.testing.assert_array_equal(
            marker_sparse.pair_idx,
            csr_mask_array_fixture.indices)

        marker_sparse = output

    ct = 0
    for i_gene in range(n_rows):
        actual = marker_sparse.get_pairs_for_gene(i_gene)
        expected = np.where(mask_array_fixture[i_gene, :][pairs_to_keep])[0]
        ct += len(actual)
        np.testing.assert_array_equal(
            actual, expected)
    assert ct > 10
