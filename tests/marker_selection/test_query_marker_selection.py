import pytest

import h5py
import json
import numpy as np
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_raw_marker_gene_lookup)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('query_selection')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def n_genes():
    return 237


@pytest.fixture(scope='module')
def taxonomy_dict_fixture():
    taxonomy_dict = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {
            'classA': ['subcA', 'subcC'],
            'classB': ['subcD', 'subcE'],
            'classC': ['subcB']
        },
        'subclass': {
            'subcA': ['a', 'd'],
            'subcB': ['b', 'e'],
            'subcC': ['c', 'f'],
            'subcD': ['g', 'i'],
            'subcE': ['h', 'j']
        },
        'cluster': {
            n:[] for n in 'abcdefghij'
        }
    }
    return taxonomy_dict


@pytest.fixture(scope='module')
def precomputed_fixture(
        taxonomy_dict_fixture,
        tmp_dir_fixture):
    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed__',
        suffix='.h5')

    taxonomy_tree = TaxonomyTree(data=taxonomy_dict_fixture)

    with h5py.File(h5_path, 'w') as dst:
        dst.create_dataset(
            'taxonomy_tree',
            data=taxonomy_tree.to_str().encode('utf-8'))

    return h5_path

@pytest.fixture(scope='module')
def pair_to_idx_fixture(
        taxonomy_dict_fixture):
    cluster_list = list(taxonomy_dict_fixture['cluster'].keys())
    cluster_list.sort()
    pair_to_idx = dict()
    pair_to_idx['cluster'] = dict()
    ct = 0
    for i0 in range(len(cluster_list)):
        n0 = cluster_list[i0]
        pair_to_idx['cluster'][n0] = dict()
        for i1 in range(i0+1, len(cluster_list), 1):
            n1 = cluster_list[i1]
            pair_to_idx['cluster'][n0][n1] = ct
            ct += 1
    return pair_to_idx

@pytest.fixture(scope='module')
def n_pairs(pair_to_idx_fixture):
    ct = 0
    for n0 in pair_to_idx_fixture['cluster']:
        for n1 in pair_to_idx_fixture['cluster'][n0]:
            ct += 1
    return ct

@pytest.fixture(scope='module')
def up_raw_fixture(
        n_genes,
        n_pairs):

    rng = np.random.default_rng(7712223)

    n_tot = n_pairs*n_genes
    data = np.zeros(n_tot, dtype=bool)
    chosen = rng.choice(
        np.arange(n_tot, dtype=int),
        n_tot//4,
        replace=False)
    data[chosen] = True
    data = data.reshape((n_pairs, n_genes))
    return data

@pytest.fixture(scope='module')
def marker_raw_fixture(
        n_genes,
        n_pairs,
        pair_to_idx_fixture):

    rng = np.random.default_rng(7712223)

    n_tot = n_pairs*n_genes
    data = np.zeros(n_tot, dtype=bool)
    chosen = rng.choice(
        np.arange(n_tot, dtype=int),
        n_tot//3,
        replace=False)
    data[chosen] = True
    data = data.reshape((n_pairs, n_genes))

    # set all markers under subclass/subcC to False
    idx = pair_to_idx_fixture['cluster']['c']['f']
    data[idx, :] = False

    return data


@pytest.fixture(scope='module')
def up_fixture(
        taxonomy_dict_fixture,
        marker_raw_fixture,
        n_genes,
        n_pairs):

    rng = np.random.default_rng(7712223)

    n_tot = n_pairs*n_genes
    data = np.zeros(n_tot, dtype=bool)
    chosen = rng.choice(
        np.arange(n_tot, dtype=int),
        n_tot//4,
        replace=False)
    data[chosen] = True
    data = data.reshape((n_pairs, n_genes))
    data = np.logical_and(data, marker_raw_fixture)

    assert data.sum() > 0
    return data

@pytest.fixture(scope='module')
def down_fixture(
        up_fixture,
        marker_raw_fixture):
    return np.logical_and(
        marker_raw_fixture,
        np.logical_not(up_fixture))


@pytest.fixture(scope='module')
def reference_marker_fixture(
        pair_to_idx_fixture,
        n_pairs,
        down_fixture,
        up_fixture,
        n_genes,
        tmp_dir_fixture):

    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_marker_cache_',
        suffix='.h5')

    gene_names = [f'gene_{ii}' for ii in range(n_genes)]

    up_by_gene = scipy_sparse.csc_array(up_fixture)
    up_by_pair = scipy_sparse.csr_array(up_fixture)
    down_by_gene = scipy_sparse.csc_array(down_fixture)
    down_by_pair = scipy_sparse.csr_array(down_fixture)

    with h5py.File(h5_path, 'w') as dst:
        dst.create_dataset(
            'n_pairs', data=n_pairs)
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_fixture).encode('utf-8'))
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names).encode('utf-8'))

        dst.create_dataset(
            'sparse_by_pair/up_pair_idx', data=up_by_pair.indptr)
        dst.create_dataset(
            'sparse_by_pair/up_gene_idx', data=up_by_pair.indices)

        dst.create_dataset(
            'sparse_by_pair/down_pair_idx', data=down_by_pair.indptr)
        dst.create_dataset(
            'sparse_by_pair/down_gene_idx', data=down_by_pair.indices)

        dst.create_dataset(
            'sparse_by_gene/up_pair_idx', data=up_by_gene.indices)
        dst.create_dataset(
            'sparse_by_gene/up_gene_idx', data=up_by_gene.indptr)

        dst.create_dataset(
            'sparse_by_gene/down_pair_idx', data=down_by_gene.indices)
        dst.create_dataset(
            'sparse_by_gene/down_gene_idx', data=down_by_gene.indptr)

    return h5_path

def test_query_marker_function(
        precomputed_fixture,
        reference_marker_fixture,
        pair_to_idx_fixture,
        marker_raw_fixture):

    with h5py.File(precomputed_fixture, 'r') as src:
        taxonomy_tree = TaxonomyTree(
            data=json.loads(src['taxonomy_tree'][()].decode('utf-8')))

    with h5py.File(reference_marker_fixture, 'r') as src:
        query_gene_names = json.loads(src['gene_names'][()].decode('utf-8'))

    gene_to_idx = {
        n:ii for ii, n in enumerate(query_gene_names)}

    marker_lookup = create_raw_marker_gene_lookup(
        input_cache_path=reference_marker_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=30,
        n_processors=3,
        behemoth_cutoff=10000)

    # we artificially set all markers under subclass/subcC to False
    assert len(marker_lookup['subclass/subcC']) == 0
    log = marker_lookup['log']
    subcC_log = log['subclass/subcC']
    assert subcC_log['n_zero'] == 1
    assert subcC_log['lt_5'] == 1
    assert subcC_log['lt_15'] == 1
    assert subcC_log['lt_30'] == 1

    # make sure classC was sckipped because there were no
    # interesting choices to make (only one child)
    classC_log = log['class/classC']
    assert 'Skipping' in classC_log['msg']

    # make sure all other parents were not ckipped
    for parent in taxonomy_tree.all_parents:
        if parent is not None:
            k = f"{parent[0]}/{parent[1]}"
        else:
            k = 'None'
        if k == 'class/classC':
            continue
        assert 'msg' not in log[k]

    # make sure that chosen genes actually are markers
    # for the parent in question
    for parent in taxonomy_tree.all_parents:
        these_leaves = taxonomy_tree.leaves_to_compare(parent)
        idx_arr = np.array(
            [pair_to_idx_fixture[l[0]][l[1]][l[2]]
             for l in these_leaves])
        if parent is None:
            parent_key = 'None'
        else:
            parent_key = f'{parent[0]}/{parent[1]}'
        actual_genes = marker_lookup[parent_key]
        if len(actual_genes) > 0:
            for g in actual_genes:
                g_idx = gene_to_idx[g]
                assert marker_raw_fixture[idx_arr, :][:, g_idx].sum() > 0
        else:
            if len(idx_arr) > 0:
                assert marker_raw_fixture[idx_arr, :].sum() == 0
