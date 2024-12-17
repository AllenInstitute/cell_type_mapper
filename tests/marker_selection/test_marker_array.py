import pytest

import copy
import h5py
import json
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.marker_selection.marker_array_utils import (
    thin_marker_gene_array_by_gene)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('marker_array'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_genes():
    return 114


@pytest.fixture
def n_cols():
    return 229


@pytest.fixture
def pair_to_idx_fixture():
    pair_to_idx = dict()
    pair_to_idx['level1'] = {
        'aa': {'bb': 0, 'cc': 1},
        'dd': {'ee': 2, 'ff': 3}}
    pair_to_idx['level2'] = {
        'a': {'b': 4, 'c': 5, 'd': 6},
        'e': {'f': 7, 'g': 8}}
    return pair_to_idx


@pytest.fixture
def is_marker_fixture(n_genes, n_cols):
    rng = np.random.default_rng(7401923)
    data = rng.integers(0, 2, (n_genes, n_cols), dtype=bool)
    return data


@pytest.fixture
def up_reg_fixture(n_genes, n_cols):
    rng = np.random.default_rng(1234567)
    data = rng.integers(0, 2, (n_genes, n_cols), dtype=bool)
    return data


@pytest.fixture
def gene_names_fixture(n_genes):
    return [f"g_{ii}" for ii in range(n_genes)]


@pytest.fixture
def up_reg_truth(up_reg_fixture, is_marker_fixture):

    up_reg = np.logical_and(
        up_reg_fixture,
        is_marker_fixture)

    return up_reg


@pytest.fixture
def down_reg_truth(up_reg_fixture, is_marker_fixture):

    down_reg = np.logical_and(
        np.logical_not(up_reg_fixture),
        is_marker_fixture)

    return down_reg


@pytest.fixture
def backed_array_fixture(
        tmp_dir_fixture,
        down_reg_truth,
        up_reg_truth,
        pair_to_idx_fixture,
        gene_names_fixture,
        n_genes,
        n_cols):

    h5_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'))

    up_by_gene = scipy_sparse.csr_array(up_reg_truth)
    up_by_pair = scipy_sparse.csc_array(up_reg_truth)

    down_by_gene = scipy_sparse.csr_array(down_reg_truth)
    down_by_pair = scipy_sparse.csc_array(down_reg_truth)

    with h5py.File(h5_path, 'a') as dst:
        dst.create_dataset('n_pairs', data=n_cols)
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_fixture).encode('utf-8'))

        by_gene_grp = dst.create_group('sparse_by_gene')
        by_gene_grp.create_dataset(
            'up_gene_idx', data=up_by_gene.indptr)
        by_gene_grp.create_dataset(
            'up_pair_idx', data=up_by_gene.indices)
        by_gene_grp.create_dataset(
            'down_gene_idx', data=down_by_gene.indptr)
        by_gene_grp.create_dataset(
            'down_pair_idx', data=down_by_gene.indices)

        by_pair_grp = dst.create_group('sparse_by_pair')
        by_pair_grp.create_dataset(
            'up_pair_idx', data=up_by_pair.indptr)
        by_pair_grp.create_dataset(
            'up_gene_idx', data=up_by_pair.indices)
        by_pair_grp.create_dataset(
            'down_pair_idx', data=down_by_pair.indptr)
        by_pair_grp.create_dataset(
            'down_gene_idx', data=down_by_pair.indices)

    return h5_path


def test_marker_gene_names(
        backed_array_fixture,
        gene_names_fixture,
        n_genes):

    arr = MarkerGeneArray.from_cache_path(cache_path=backed_array_fixture)
    assert arr.gene_names == gene_names_fixture
    assert arr.n_genes == n_genes


def test_idx_of_pair_error(backed_array_fixture):
    arr = MarkerGeneArray.from_cache_path(cache_path=backed_array_fixture)
    with pytest.raises(RuntimeError, match="not under taxonomy level level1"):
        arr.idx_of_pair(level='level1', node1='garbage', node2='other')

    with pytest.raises(RuntimeError, match="not a valid taxonomy pair"):
        arr.idx_of_pair(level='level1', node1='aa', node2='ff')


def test_idx_of_pair_idx(backed_array_fixture, pair_to_idx_fixture):
    arr = MarkerGeneArray.from_cache_path(cache_path=backed_array_fixture)
    for level in pair_to_idx_fixture:
        for node1 in pair_to_idx_fixture[level]:
            for node2 in pair_to_idx_fixture[level][node1]:
                actual = arr.idx_of_pair(
                    level=level,
                    node1=node1,
                    node2=node2)
                expected = pair_to_idx_fixture[level][node1][node2]
                assert actual == expected


def test_marker_mask_from_gene_idx(
        backed_array_fixture,
        down_reg_truth,
        up_reg_truth,
        is_marker_fixture,
        n_genes):
    arr = MarkerGeneArray.from_cache_path(cache_path=backed_array_fixture)

    for i_gene in range(n_genes):
        (actual_marker,
         actual_up) = arr.marker_mask_from_gene_idx(gene_idx=i_gene)
        np.testing.assert_array_equal(
            actual_marker,
            is_marker_fixture[i_gene, :])
        np.testing.assert_array_equal(
            actual_up,
            up_reg_truth[i_gene, :])


def test_marker_mask_from_pair_idx(
        backed_array_fixture,
        is_marker_fixture,
        up_reg_truth,
        n_cols):

    arr = MarkerGeneArray.from_cache_path(
        cache_path=backed_array_fixture)

    for i_col in range(n_cols):
        (actual_marker,
         actual_up) = arr.marker_mask_from_pair_idx(pair_idx=i_col)

        expected_marker = is_marker_fixture[:, i_col]
        expected_up = up_reg_truth[:, i_col]

        np.testing.assert_array_equal(
            actual_marker,
            expected_marker)

        np.testing.assert_array_equal(
            actual_up,
            expected_up)

        np.testing.assert_array_equal(
            arr.up_mask_from_pair_idx(pair_idx=i_col),
            np.logical_and(expected_marker, expected_up))

        np.testing.assert_array_equal(
            arr.down_mask_from_pair_idx(pair_idx=i_col),
            np.logical_and(expected_marker, ~expected_up))


@pytest.mark.parametrize('to_other', [True, False])
def test_marker_downsample_genes(
        backed_array_fixture,
        is_marker_fixture,
        up_reg_truth,
        n_cols,
        n_genes,
        gene_names_fixture,
        to_other):

    arr = MarkerGeneArray.from_cache_path(
        cache_path=backed_array_fixture)

    assert arr.n_genes == n_genes
    assert arr.n_pairs == n_cols

    rng = np.random.default_rng(55123)
    subsample = rng.choice(np.arange(n_genes), 8, replace=False)
    if to_other:
        arr0 = arr
        np.testing.assert_array_equal(
            arr0.up_by_gene.indices,
            arr.up_by_gene.indices
        )
        arr = arr.downsample_genes_to_other(gene_idx_array=subsample)
        assert arr is not arr0
        assert not (
           np.array_equal(
               arr0.up_by_gene.indices,
               arr.up_by_gene.indices)
        )
    else:
        arr.downsample_genes(gene_idx_array=subsample)

    assert arr.n_pairs == n_cols
    assert arr.n_genes == 8
    assert arr.gene_names == [gene_names_fixture[ii] for ii in subsample]
    for ii, i_gene in enumerate(subsample):
        (marker,
         up) = arr.marker_mask_from_gene_idx(ii)

        expected_marker = is_marker_fixture[i_gene, :]
        expected_up = up_reg_truth[i_gene, :]

        np.testing.assert_array_equal(
            marker,
            expected_marker)

        np.testing.assert_array_equal(
            up,
            expected_up)

    for i_col in range(n_cols):
        (marker,
         up) = arr.marker_mask_from_pair_idx(i_col)

        expected_marker = is_marker_fixture[subsample, i_col]
        expected_up = up_reg_truth[subsample, i_col]

        np.testing.assert_array_equal(
            marker,
            expected_marker)

        np.testing.assert_array_equal(
            up,
            expected_up)

        np.testing.assert_array_equal(
            arr.up_mask_from_pair_idx(i_col),
            np.logical_and(expected_marker, expected_up))

        np.testing.assert_array_equal(
            arr.down_mask_from_pair_idx(i_col),
            np.logical_and(expected_marker, ~expected_up))


def test_downsampling_by_taxon_pairs(
       backed_array_fixture,
       pair_to_idx_fixture):

    base_array = MarkerGeneArray.from_cache_path(
            cache_path=backed_array_fixture)
    pairs_to_keep = [('level2', 'e', 'g'), ('level1', 'dd', 'ff'),
                     ('level2', 'a', 'c')]

    test_array = MarkerGeneArray.from_cache_path(
            cache_path=backed_array_fixture)

    test_array = test_array.downsample_pairs_to_other(
            only_keep_pairs=pairs_to_keep)

    assert test_array.n_genes == base_array.n_genes
    assert test_array.n_pairs == len(pairs_to_keep)
    assert test_array.n_pairs < base_array.n_pairs
    assert test_array.gene_names == base_array.gene_names
    for ii, pair in enumerate(pairs_to_keep):
        base_idx = base_array.idx_of_pair(
            pair[0],
            pair[1],
            pair[2])
        base_m, base_u = base_array.marker_mask_from_pair_idx(base_idx)
        test_m, test_u = test_array.marker_mask_from_pair_idx(ii)
        np.testing.assert_array_equal(test_m, base_m)
        np.testing.assert_array_equal(test_u, base_u)

        np.testing.assert_array_equal(
            base_array.up_mask_from_pair_idx(base_idx),
            test_array.up_mask_from_pair_idx(ii))

        np.testing.assert_array_equal(
            base_array.down_mask_from_pair_idx(base_idx),
            test_array.down_mask_from_pair_idx(ii))

        assert test_array.idx_of_pair(
                    pair[0],
                    pair[1],
                    pair[2]) == ii

    for level in pair_to_idx_fixture:
        for node1 in pair_to_idx_fixture[level]:
            for node2 in pair_to_idx_fixture[level][node1]:
                pair = (level, node1, node2)
                if pair not in pairs_to_keep:
                    with pytest.raises(RuntimeError, match='taxonomy'):
                        test_array.idx_of_pair(
                            pair[0], pair[1], pair[2])
                else:
                    test_array.idx_of_pair(
                        pair[0], pair[1], pair[2])


def test_downsampling_by_taxon_pairs_other(
       backed_array_fixture,
       pair_to_idx_fixture,
       n_genes):

    base_array = MarkerGeneArray.from_cache_path(
            cache_path=backed_array_fixture)
    pairs_to_keep = [('level2', 'e', 'g'), ('level1', 'dd', 'ff'),
                     ('level2', 'a', 'c')]
    test_array = base_array.downsample_pairs_to_other(
        only_keep_pairs=pairs_to_keep)

    assert test_array is not base_array

    assert test_array.n_genes == base_array.n_genes
    assert test_array.n_pairs == len(pairs_to_keep)
    assert test_array.n_pairs < base_array.n_pairs
    assert test_array.gene_names == base_array.gene_names
    for ii, pair in enumerate(pairs_to_keep):
        base_idx = base_array.idx_of_pair(
            pair[0],
            pair[1],
            pair[2])

        assert base_idx != ii

        base_m, base_u = base_array.marker_mask_from_pair_idx(base_idx)
        test_m, test_u = test_array.marker_mask_from_pair_idx(ii)
        np.testing.assert_array_equal(test_m, base_m)
        np.testing.assert_array_equal(test_u, base_u)

        np.testing.assert_array_equal(
            base_array.up_mask_from_pair_idx(base_idx),
            test_array.up_mask_from_pair_idx(ii))

        np.testing.assert_array_equal(
            base_array.down_mask_from_pair_idx(base_idx),
            test_array.down_mask_from_pair_idx(ii))

        assert test_array.idx_of_pair(
                    pair[0],
                    pair[1],
                    pair[2]) == ii

    for level in pair_to_idx_fixture:
        for node1 in pair_to_idx_fixture[level]:
            for node2 in pair_to_idx_fixture[level][node1]:
                pair = (level, node1, node2)
                if pair not in pairs_to_keep:
                    with pytest.raises(RuntimeError, match='taxonomy'):
                        test_array.idx_of_pair(
                            pair[0], pair[1], pair[2])
                else:
                    test_array.idx_of_pair(
                        pair[0], pair[1], pair[2])

    # make sure base array data was not changed
    with h5py.File(backed_array_fixture, 'r') as expected:
        assert base_array.n_pairs == expected['n_pairs'][()]
        expected_lookup = json.loads(
            expected['pair_to_idx'][()].decode('utf-8'))
        expected_gene_names = json.loads(
            expected['gene_names'][()].decode('utf-8'))
        assert base_array.taxonomy_pair_to_idx == expected_lookup
        assert base_array.gene_names == expected_gene_names

        np.testing.assert_array_equal(
            expected['sparse_by_pair/up_gene_idx'][()],
            base_array.up_by_pair.indices)
        np.testing.assert_array_equal(
            expected['sparse_by_pair/up_pair_idx'][()],
            base_array.up_by_pair.indptr)
        np.testing.assert_array_equal(
            expected['sparse_by_pair/down_gene_idx'][()],
            base_array.down_by_pair.indices)
        np.testing.assert_array_equal(
            expected['sparse_by_pair/down_pair_idx'][()],
            base_array.down_by_pair.indptr)
        for sparse in (base_array.down_by_pair,
                       base_array.up_by_pair):
            assert len(sparse.indptr) == base_array.n_pairs+1
            assert sparse.indptr[-1] == len(sparse.indices)

        np.testing.assert_array_equal(
            expected['sparse_by_gene/up_gene_idx'][()],
            base_array.up_by_gene.indptr)
        np.testing.assert_array_equal(
            expected['sparse_by_gene/up_pair_idx'][()],
            base_array.up_by_gene.indices)
        np.testing.assert_array_equal(
            expected['sparse_by_gene/down_gene_idx'][()],
            base_array.down_by_gene.indptr)
        np.testing.assert_array_equal(
            expected['sparse_by_gene/down_pair_idx'][()],
            base_array.down_by_gene.indices)
        for sparse in (base_array.down_by_gene,
                       base_array.up_by_gene):
            assert len(sparse.indptr) == n_genes+1
            assert sparse.indptr[-1] == len(sparse.indices)


def test_thin_array_by_gene(
        backed_array_fixture,
        gene_names_fixture,
        up_reg_truth,
        down_reg_truth,
        n_genes,
        tmp_dir_fixture):

    arr = MarkerGeneArray.from_cache_path(cache_path=backed_array_fixture)

    rng = np.random.default_rng(553321)
    valid_query_genes = rng.choice(
        gene_names_fixture,
        n_genes//3,
        replace=False)
    query_genes = list(valid_query_genes) + [f'junk_{ii}' for ii in range(15)]
    rng.shuffle(query_genes)

    arr = thin_marker_gene_array_by_gene(
        marker_gene_array=arr,
        query_gene_names=query_genes,
        tmp_dir=tmp_dir_fixture)

    assert arr.n_genes > 0
    assert arr.n_genes == len(valid_query_genes)

    assert set(arr.gene_names) == set(valid_query_genes)

    query_idx = np.array([ii for ii, g in enumerate(gene_names_fixture)
                          if g in valid_query_genes])

    new_up = up_reg_truth[query_idx, :]
    new_down = down_reg_truth[query_idx, :]

    up_csr = scipy_sparse.csr_array(new_up)
    np.testing.assert_array_equal(
        arr.up_by_gene.indptr, up_csr.indptr)
    np.testing.assert_array_equal(
        arr.up_by_gene.indices, up_csr.indices)

    up_csc = scipy_sparse.csc_array(new_up)
    np.testing.assert_array_equal(
        arr.up_by_pair.indptr, up_csc.indptr)
    np.testing.assert_array_equal(
        arr.up_by_pair.indices, up_csc.indices)

    down_csr = scipy_sparse.csr_array(new_down)
    np.testing.assert_array_equal(
        arr.down_by_gene.indptr, down_csr.indptr)
    np.testing.assert_array_equal(
        arr.down_by_gene.indices, down_csr.indices)

    down_csc = scipy_sparse.csc_array(new_down)
    np.testing.assert_array_equal(
        arr.down_by_pair.indptr, down_csc.indptr)
    np.testing.assert_array_equal(
        arr.down_by_pair.indices, down_csc.indices)


@pytest.mark.parametrize('downsample_genes', [True, False])
def test_thin_array_by_gene_on_load(
        backed_array_fixture,
        gene_names_fixture,
        up_reg_truth,
        down_reg_truth,
        n_genes,
        tmp_dir_fixture,
        downsample_genes):

    rng = np.random.default_rng(553321)

    if downsample_genes:
        valid_query_genes = rng.choice(
            gene_names_fixture,
            n_genes//3,
            replace=False)
    else:
        valid_query_genes = copy.deepcopy(gene_names_fixture)
    query_genes = list(valid_query_genes) + [f'junk_{ii}' for ii in range(15)]
    rng.shuffle(query_genes)

    arr = MarkerGeneArray.from_cache_path(
            cache_path=backed_array_fixture,
            query_gene_names=query_genes,
            tmp_dir=tmp_dir_fixture)

    assert arr.n_genes > 0
    assert arr.n_genes == len(valid_query_genes)

    assert set(arr.gene_names) == set(valid_query_genes)

    query_idx = np.array([ii for ii, g in enumerate(gene_names_fixture)
                          if g in valid_query_genes])

    new_up = up_reg_truth[query_idx, :]
    new_down = down_reg_truth[query_idx, :]

    up_csr = scipy_sparse.csr_array(new_up)
    np.testing.assert_array_equal(
        arr.up_by_gene.indptr, up_csr.indptr)
    np.testing.assert_array_equal(
        arr.up_by_gene.indices, up_csr.indices)

    up_csc = scipy_sparse.csc_array(new_up)
    np.testing.assert_array_equal(
        arr.up_by_pair.indptr, up_csc.indptr)
    np.testing.assert_array_equal(
        arr.up_by_pair.indices, up_csc.indices)

    down_csr = scipy_sparse.csr_array(new_down)
    np.testing.assert_array_equal(
        arr.down_by_gene.indptr, down_csr.indptr)
    np.testing.assert_array_equal(
        arr.down_by_gene.indices, down_csr.indices)

    down_csc = scipy_sparse.csc_array(new_down)
    np.testing.assert_array_equal(
        arr.down_by_pair.indptr, down_csc.indptr)
    np.testing.assert_array_equal(
        arr.down_by_pair.indices, down_csc.indices)
