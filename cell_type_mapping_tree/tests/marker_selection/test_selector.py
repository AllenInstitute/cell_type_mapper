import pytest

import copy
from itertools import combinations
import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.utils.multiprocessing_utils import (
    DummyLock)

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)

from hierarchical_mapping.marker_selection.marker_array import (
    MarkerGeneArray)

from hierarchical_mapping.marker_selection.selection import (
    select_marker_genes_v2)

from hierarchical_mapping.marker_selection.selection_pipeline import (
    _marker_selection_worker)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('selector'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def taxonomy_tree_fixture():

    rng = np.random.default_rng(77123)

    tree = dict()
    tree['hierarchy'] = ['class', 'subclass', 'cluster']
    tree['class']  = {
        'aa': ['a', 'b', 'c', 'd'],
        'bb': ['e'],
        'cc': ['f', 'g', 'h']}

    cluster_list = []
    name_ct = 0
    tree['subclass'] = dict()
    for subclass in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        n_clusters = rng.integers(3, 8)
        tree['subclass'][subclass] = []
        for ii in range(n_clusters):
            n = f'cluster_{name_ct}'
            cluster_list.append(n)
            tree['subclass'][subclass].append(n)
            name_ct += 1

    tree['cluster'] = dict()
    c0 = 0
    for cluster in cluster_list:
        rows = rng.integers(2, 6)
        tree['cluster'][cluster] = []
        for ii in range(rows):
            tree['cluster'][cluster].append(c0+ii)
        c0 += rows

    return tree

@pytest.fixture
def pair_to_idx_fixture(
        taxonomy_tree_fixture):

    leaf = taxonomy_tree_fixture['hierarchy'][-1]
    cluster_list = list(taxonomy_tree_fixture[leaf].keys())
    cluster_list.sort()
    pair_to_idx = dict()
    pair_to_idx['cluster'] = dict()
    for idx, pair in enumerate(combinations(cluster_list, 2)):
        assert pair[0] < pair[1]
        if pair[0] not in pair_to_idx['cluster']:
            pair_to_idx['cluster'][pair[0]] = dict()
        pair_to_idx['cluster'][pair[0]][pair[1]] = idx
    pair_to_idx['n_pairs'] = idx + 1
    return pair_to_idx


@pytest.fixture
def gene_names_fixture():
    n_genes = 47
    gene_names = [f'g_{ii}' for ii in range(n_genes)]
    return gene_names


@pytest.fixture
def is_marker_fixture(
        pair_to_idx_fixture,
        gene_names_fixture):
    n_pairs = pair_to_idx_fixture['n_pairs']
    n_genes = len(gene_names_fixture)
    rng = np.random.default_rng(876543)
    data = rng.integers(0, 2, (n_genes, n_pairs), dtype=bool)
    return data


@pytest.fixture
def up_reg_fixture(
        pair_to_idx_fixture,
        gene_names_fixture):
    n_pairs = pair_to_idx_fixture['n_pairs']
    n_genes = len(gene_names_fixture)
    rng = np.random.default_rng(25789)
    data = rng.integers(0, 2, (n_genes, n_pairs), dtype=bool)
    return data


@pytest.fixture
def marker_cache_fixture(
         tmp_dir_fixture,
         is_marker_fixture,
         up_reg_fixture,
         gene_names_fixture,
         pair_to_idx_fixture):

    out_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='markers_',
            suffix='.h5'))

    n_rows = len(gene_names_fixture)
    n_cols = pair_to_idx_fixture['n_pairs']

    is_marker = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    up_reg = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    for i_row in range(n_rows):
        is_marker.set_row(i_row, is_marker_fixture[i_row, :])
        up_reg.set_row(i_row, up_reg_fixture[i_row, :])

    is_marker.write_to_h5(
        h5_path=out_path,
        h5_group='markers')

    up_reg.write_to_h5(
        h5_path=out_path,
        h5_group='up_regulated')

    with h5py.File(out_path, 'a') as dst:
        pair_to_idx = copy.deepcopy(pair_to_idx_fixture)
        pair_to_idx.pop('n_pairs')
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx).encode('utf-8'))
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'n_pairs',
            data=n_cols)

    return out_path


@pytest.fixture
def blank_marker_cache_fixture(
         tmp_dir_fixture,
         gene_names_fixture,
         pair_to_idx_fixture):
    """
    Case where there are no marker genes
    """
    out_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='blank_markers_',
            suffix='.h5'))

    n_rows = len(gene_names_fixture)
    n_cols = pair_to_idx_fixture['n_pairs']

    is_marker = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    up_reg = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    is_marker.write_to_h5(
        h5_path=out_path,
        h5_group='markers')

    up_reg.write_to_h5(
        h5_path=out_path,
        h5_group='up_regulated')

    with h5py.File(out_path, 'a') as dst:
        pair_to_idx = copy.deepcopy(pair_to_idx_fixture)
        pair_to_idx.pop('n_pairs')
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx).encode('utf-8'))
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'n_pairs',
            data=n_cols)

    return out_path


def test_selecting_from_blank_markers(
        gene_names_fixture,
        taxonomy_tree_fixture,
        blank_marker_cache_fixture):

    marker_array = MarkerGeneArray(
        cache_path=blank_marker_cache_fixture)

    marker_genes = select_marker_genes_v2(
        marker_gene_array=marker_array,
        query_gene_names=gene_names_fixture,
        taxonomy_tree=taxonomy_tree_fixture,
        parent_node=None,
        n_per_utility=5)

    assert marker_genes == []

@pytest.mark.parametrize("behemoth_cutoff", [1000000, 5])
def test_selection_worker_smoke(
         marker_cache_fixture,
         gene_names_fixture,
         taxonomy_tree_fixture,
         behemoth_cutoff):
    """
    Run a smoketest of _marker_selection_worker
    """
    rng = np.random.default_rng(2231)
    query_gene_names = rng.choice(gene_names_fixture, 40, replace=False)
    output_dict = dict()
    input_lock = DummyLock()

    parent_list = [None,
                   ('subclass', 'e'),
                   ('class', 'aa'),
                   ('class', 'bb')]

    for parent in parent_list:
        _marker_selection_worker(
            marker_cache_path=marker_cache_fixture,
            query_gene_names=query_gene_names,
            taxonomy_tree=taxonomy_tree_fixture,
            parent_node=parent,
            behemoth_cutoff=behemoth_cutoff,
            n_per_utility=5,
            output_dict=output_dict,
            input_lock=input_lock)

    for parent in parent_list:
        if parent == ('class', 'bb'):
            assert len(output_dict[parent]) == 0
        else:
            assert len(output_dict[parent]) > 0
            for g in output_dict[parent]:
                assert g in gene_names_fixture
