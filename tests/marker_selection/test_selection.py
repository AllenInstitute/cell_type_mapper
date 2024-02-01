import pytest

import h5py
import json
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.marker_selection.selection import (
    recalculate_utility_array,
    recalculate_utility_array_batch,
    _get_taxonomy_idx)

from cell_type_mapper.diff_exp.markers import (
    add_sparse_by_gene_markers_to_file)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('selection_module'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_genes():
    return 14

@pytest.fixture
def n_cols():
    return 31

@pytest.fixture
def taxonomy_tree_fixture():
    tree = dict()
    tree['hierarchy'] = ['level1', 'level2', 'cluster']
    tree['level1'] = {'a': ['aa', 'bb'],
                      'b': ['cc', 'dd']}
    tree['level2'] = {
        'aa': ['1', '2'],
        'bb': ['3', '4', '5'],
        'cc': ['6'],
        'dd': ['7', '8']}
    tree['cluster'] = {str(ii): [ii*10] for ii in range(1, 9)}

    return tree

@pytest.fixture
def pair_to_idx_fixture():
    pair_to_idx = dict()
    pair_to_idx['cluster'] = dict()
    pair_to_idx['cluster']['1'] = {
        '2': 0,
        '3': 1,
        '4': 5,
        '5': 6,
        '6': 7,
        '7': 8,
        '8': 9}
    pair_to_idx['cluster']['2'] = {
        '3': 10,
        '4': 11,
        '5': 12,
        '6': 13,
        '7': 14,
        '8': 15}

    pair_to_idx['cluster']['3'] = {
        '4': 16,
        '5': 17,
        '6': 18,
        '7': 19,
        '8': 20}

    pair_to_idx['cluster']['4'] = {
        '5': 21,
        '6': 22,
        '7': 23,
        '8': 24}

    pair_to_idx['cluster']['5'] = {
        '6': 25,
        '7': 26,
        '8': 27}

    pair_to_idx['cluster']['6'] = {
        '7': 28,
        '8': 29}

    pair_to_idx['cluster']['7'] = {
        '8': 30}

    return pair_to_idx

@pytest.fixture
def is_marker_fixture(n_genes, n_cols):
    data = np.zeros((n_genes, n_cols), dtype=bool)
    data[2, 4] = True
    data[3, 4] = True
    data[11, 4] = True
    return data

@pytest.fixture
def up_reg_fixture(n_genes, n_cols):
    data = np.zeros((n_genes, n_cols), dtype=bool)
    data[2, 4] = True
    data[11, 4] = True
    return data

@pytest.fixture
def gene_names_fixture(n_genes):
    return [f"g_{ii}" for ii in range(n_genes)]

@pytest.fixture
def marker_cache_fixture(
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

    up_regulated = np.logical_and(
        is_marker_fixture,
        up_reg_fixture)
    down_regulated = np.logical_and(
        is_marker_fixture,
        np.logical_not(up_reg_fixture))

    csc_up = scipy_sparse.csc_array(up_regulated)
    csc_down = scipy_sparse.csc_array(down_regulated)

    with h5py.File(h5_path, 'a') as dst:
        dst.create_dataset('n_pairs', data=n_cols)
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_fixture).encode('utf-8'))

        grp = dst.create_group('sparse_by_pair')
        grp.create_dataset(
            'up_gene_idx',
            data=csc_up.indices)
        grp.create_dataset(
            'up_pair_idx',
            data=csc_up.indptr)
        grp.create_dataset(
            'down_gene_idx',
            data=csc_down.indices)
        grp.create_dataset(
            'down_pair_idx',
            data=csc_down.indptr)

    add_sparse_by_gene_markers_to_file(
        h5_path=h5_path,
        n_genes=len(gene_names_fixture),
        max_gb=3,
        tmp_dir=tmp_dir_fixture)

    return h5_path


def test_recalculate_utilty_array(
        marker_cache_fixture,
        n_genes):
    arr = MarkerGeneArray.from_cache_path(
        cache_path=marker_cache_fixture)
    util = np.zeros(n_genes, dtype=int)
    util = recalculate_utility_array(
        utility_array=util,
        marker_gene_array=arr,
        pair_idx=4,
        sign=-1)
    expected = np.zeros(n_genes, dtype=int)
    expected[3] = -1
    np.testing.assert_array_equal(expected, util)

    util = np.zeros(n_genes, dtype=int)
    util = recalculate_utility_array(
        utility_array=util,
        marker_gene_array=arr,
        pair_idx=4,
        sign=1)
    expected = np.zeros(n_genes, dtype=int)
    expected[2] = -1
    expected[11] = -1
    np.testing.assert_array_equal(expected, util)

    with pytest.raises(RuntimeError, match="Unclear how to interpret sign"):
        recalculate_utility_array(
            utility_array=util,
            marker_gene_array=arr,
            pair_idx=4,
            sign=3)

def test_recalculate_utility_array_batch(
        marker_cache_fixture,
        n_genes):
    """
    Test method that runs a bunch of utility array recalculations at once
    """
    rng = np.random.default_rng(67112)
    marker_arr = MarkerGeneArray.from_cache_path(
        cache_path=marker_cache_fixture)
    util = rng.integers(1000, 5000, n_genes)

    n_adjustments = 100
    sign_arr = rng.choice((-1, 1), n_adjustments, replace=True)
    pair_arr = rng.choice(
        np.arange(marker_arr.n_pairs), n_adjustments, replace=True)
    expected = np.copy(util)
    for sign, pair in zip(sign_arr, pair_arr):
        expected = recalculate_utility_array(
            marker_gene_array=marker_arr,
            utility_array=expected,
            pair_idx=pair,
            sign=sign)

    actual = np.copy(util)
    actual = recalculate_utility_array_batch(
        marker_gene_array=marker_arr,
        utility_array=actual,
        pair_batch=pair_arr,
        sign_batch=sign_arr)

    np.testing.assert_array_equal(
        actual,
        expected)


def test_get_taxonomy_idx(
        taxonomy_tree_fixture,
        marker_cache_fixture):

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_fixture)

    arr = MarkerGeneArray.from_cache_path(
        cache_path=marker_cache_fixture)
    np.testing.assert_array_equal(
        _get_taxonomy_idx(
            taxonomy_tree=taxonomy_tree,
            parent_node=('level1', 'b'),
            marker_gene_array=arr),
        np.array([28, 29]))

    np.testing.assert_array_equal(
        _get_taxonomy_idx(
            taxonomy_tree=taxonomy_tree,
            parent_node=('level2', 'aa'),
            marker_gene_array=arr),
        np.array([0,]))
