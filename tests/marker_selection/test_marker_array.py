import pytest

import h5py
import json
import numpy as np
import pathlib
import shutil
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.binary_array.binary_array import (
    BinarizedBooleanArray)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.diff_exp.sparse_markers_by_pair import (
    add_sparse_markers_by_pair_to_h5)


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
def backed_array_fixture(
        tmp_dir_fixture,
        is_marker_fixture,
        up_reg_fixture,
        pair_to_idx_fixture,
        gene_names_fixture,
        n_genes,
        n_cols):

    h5_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'))

    h5_path.unlink()

    marker = BinarizedBooleanArray(
        n_rows=n_genes,
        n_cols=n_cols)
    for i_row in range(n_genes):
        marker.set_row(i_row, is_marker_fixture[i_row, :])
    marker.write_to_h5(h5_path, h5_group='markers')

    up_reg = BinarizedBooleanArray(
        n_rows=n_genes,
        n_cols=n_cols)
    for i_row in range(n_genes):
        up_reg.set_row(i_row, up_reg_fixture[i_row, :])
    up_reg.write_to_h5(h5_path, h5_group='up_regulated')

    with h5py.File(h5_path, 'a') as dst:
        dst.create_dataset('n_pairs', data=n_cols)
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_fixture).encode('utf-8'))

    return h5_path


@pytest.fixture
def backed_array_fixture_with_sparse(
        tmp_dir_fixture,
        backed_array_fixture):

    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='markers_with_sparse_',
        suffix='.h5')

    shutil.copy(
        src=backed_array_fixture,
        dst=h5_path)

    add_sparse_markers_by_pair_to_h5(h5_path)
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
        is_marker_fixture,
        up_reg_fixture,
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
            up_reg_fixture[i_gene, :])

@pytest.mark.parametrize('downsample_pairs', [True, False])
def test_use_sparse(
       backed_array_fixture,
       backed_array_fixture_with_sparse,
       downsample_pairs):

    if downsample_pairs:
        pairs_to_keep = [('level2', 'e', 'g'), ('level1', 'dd', 'ff'),
                         ('level2', 'a', 'c')]
    else:
        pairs_to_keep = None

    base = MarkerGeneArray.from_cache_path(
            cache_path=backed_array_fixture,
            only_keep_pairs=pairs_to_keep)
    test = MarkerGeneArray.from_cache_path(
            cache_path=backed_array_fixture_with_sparse,
            only_keep_pairs=pairs_to_keep)

    for i_pair in range(base.n_pairs):
        np.testing.assert_array_equal(
            base._up_mask_from_pair_idx_use_full(i_pair),
            test._up_mask_from_pair_idx_use_sparse(i_pair))
        np.testing.assert_array_equal(
            base._down_mask_from_pair_idx_use_full(i_pair),
            test._down_mask_from_pair_idx_use_sparse(i_pair))

        np.testing.assert_array_equal(
            base.up_mask_from_pair_idx(i_pair),
            test.up_mask_from_pair_idx(i_pair))
        np.testing.assert_array_equal(
            base.down_mask_from_pair_idx(i_pair),
            test.down_mask_from_pair_idx(i_pair))


@pytest.mark.parametrize('use_sparse', [True, False])
def test_marker_mask_from_pair_idx(
        backed_array_fixture,
        backed_array_fixture_with_sparse,
        is_marker_fixture,
        up_reg_fixture,
        n_cols,
        use_sparse):

    if use_sparse:
        pth = backed_array_fixture_with_sparse
    else:
        pth = backed_array_fixture

    arr = MarkerGeneArray.from_cache_path(cache_path=pth)

    if use_sparse:
        arr._up_marker_sparse is not None
        arr._down_marker_sparse is not None
    else:
        arr._up_marker_sparse is None
        arr._down_marker_sparse is None

    for i_col in range(n_cols):
        (actual_marker,
         actual_up) = arr.marker_mask_from_pair_idx(pair_idx=i_col)

        expected_marker = is_marker_fixture[:, i_col]
        expected_up = up_reg_fixture[:, i_col]

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

@pytest.mark.parametrize('use_sparse', [True, False])
def test_marker_downsample_genes(
        backed_array_fixture,
        backed_array_fixture_with_sparse,
        is_marker_fixture,
        up_reg_fixture,
        n_cols,
        n_genes,
        gene_names_fixture,
        use_sparse):

    if use_sparse:
        pth = backed_array_fixture_with_sparse
    else:
        pth = backed_array_fixture

    arr = MarkerGeneArray.from_cache_path(cache_path=pth)

    if use_sparse:
        arr._up_marker_sparse is not None
        arr._down_marker_sparse is not None
    else:
        arr._up_marker_sparse is None
        arr._down_marker_sparse is None

    assert arr.n_genes == n_genes
    assert arr.n_pairs == n_cols

    rng = np.random.default_rng(55123)
    subsample = rng.choice(np.arange(n_genes), 8, replace=False)
    arr.downsample_genes(gene_idx_array=subsample)
    assert arr.n_pairs == n_cols
    assert arr.n_genes == 8
    assert arr.gene_names == [gene_names_fixture[ii] for ii in subsample]
    for ii, i_gene in enumerate(subsample):
        (marker,
         up) = arr.marker_mask_from_gene_idx(ii)

        expected_marker = is_marker_fixture[i_gene, :]
        expected_up = up_reg_fixture[i_gene, :]

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
        expected_up = up_reg_fixture[subsample, i_col]

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


@pytest.mark.parametrize('use_sparse', [True, False])
def test_downsampling_by_taxon_pairs(
       backed_array_fixture,
       backed_array_fixture_with_sparse,
       pair_to_idx_fixture,
       use_sparse):

    if use_sparse:
        test_path = backed_array_fixture_with_sparse
    else:
        test_path = backed_array_fixture

    base_array = MarkerGeneArray.from_cache_path(
            cache_path=backed_array_fixture)
    pairs_to_keep = [('level2', 'e', 'g'), ('level1', 'dd', 'ff'),
                     ('level2', 'a', 'c')]
    test_array = MarkerGeneArray.from_cache_path(
            cache_path=test_path,
            only_keep_pairs=pairs_to_keep)

    if use_sparse:
        test_array._up_marker_sparse is not None
        test_array._down_marker_sparse is not None
    else:
        test_array._up_marker_sparse is None
        test_array._down_marker_sparse is None

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


@pytest.mark.parametrize(
    'use_sparse, copy_sparse',
    [(True, True), (True, False), (False, False)])
def test_downsampling_by_taxon_pairs_other(
       backed_array_fixture,
       backed_array_fixture_with_sparse,
       pair_to_idx_fixture,
       use_sparse,
       copy_sparse):

    if use_sparse:
        pth = backed_array_fixture_with_sparse
    else:
        pth = backed_array_fixture

    base_array = MarkerGeneArray.from_cache_path(
            cache_path=pth)
    pairs_to_keep = [('level2', 'e', 'g'), ('level1', 'dd', 'ff'),
                     ('level2', 'a', 'c')]
    test_array = base_array.downsample_pairs_to_other(
        only_keep_pairs=pairs_to_keep,
        copy_sparse=copy_sparse)

    if use_sparse:
        assert base_array._up_marker_sparse is not None
        assert base_array._down_marker_sparse is not None
    else:
        assert base_array._up_marker_sparse is None
        assert base_array._down_marker_sparse is None

    if use_sparse and copy_sparse:
        assert test_array._up_marker_sparse is not None
        assert test_array._down_marker_sparse is not None
    else:
        assert test_array._up_marker_sparse is None
        assert test_array._down_marker_sparse is None

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
        expected_markers = expected['markers/data'][()]
        np.testing.assert_array_equal(
            base_array.is_marker.data,
            expected_markers)
        assert not np.array_equal(
            test_array.is_marker.data,
            expected_markers)

        expected_up = expected['up_regulated/data'][()]
        np.testing.assert_array_equal(
            base_array.up_regulated.data,
            expected_up)
        assert not np.array_equal(
            test_array.up_regulated.data,
            expected_up)

    if use_sparse:
        with h5py.File(backed_array_fixture_with_sparse, 'r') as expected:
            np.testing.assert_array_equal(
                expected['sparse_by_pair/up_gene_idx'][()],
                base_array._up_marker_sparse.gene_idx)
            np.testing.assert_array_equal(
                expected['sparse_by_pair/up_pair_idx'][()],
                base_array._up_marker_sparse.pair_idx)
            np.testing.assert_array_equal(
                expected['sparse_by_pair/down_gene_idx'][()],
                base_array._down_marker_sparse.gene_idx)
            np.testing.assert_array_equal(
                expected['sparse_by_pair/down_pair_idx'][()],
                base_array._down_marker_sparse.pair_idx)
            for sparse in (base_array._down_marker_sparse,
                            base_array._up_marker_sparse):
                assert len(sparse.pair_idx) == base_array.n_pairs+1
                assert sparse.pair_idx[-1] == len(sparse.gene_idx)
