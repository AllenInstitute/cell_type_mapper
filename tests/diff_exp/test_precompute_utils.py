import pytest

import h5py
import json
import numpy as np

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_utils import (
    run_leaf_census,
    merge_precompute_files)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('precompute_utils_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def taxonomy_tree_fixture():
    data = {
        'hierarchy': ['class', 'cluster'],
        'class': {
            'A': ['a', 'b', 'c', 'h', 'i', 'j'],
            'B': ['d', 'e', 'f', 'g']
        },
        'cluster': {
            n:[] for n in 'abcdefghij'
        }
    }
    tree = TaxonomyTree(data=data)
    return tree


@pytest.fixture(scope='module')
def expected_census(
        tmp_dir_fixture,
        taxonomy_tree_fixture):
    leaf_list = taxonomy_tree_fixture.all_leaves
    result = dict()
    for leaf in leaf_list:
        result[leaf] = dict()
    rng = np.random.default_rng(8712311)
    row_idx = list(range(len(leaf_list)))
    for i_file in range(3):
        pth = mkstemp_clean(
                dir=tmp_dir_fixture,
                prefix='for_census_',
                suffix='.h5')

        for leaf in leaf_list:
            result[leaf][str(pth)] = rng.integers(0, 255)
        rng.shuffle(row_idx)
        cluster_to_row = {
            leaf_list[ii]: row_idx[ii] for ii in range(len(leaf_list))}
        n_cells = np.zeros(len(leaf_list), dtype=int)
        for leaf in leaf_list:
            n_cells[cluster_to_row[leaf]] = int(result[leaf][str(pth)])
        with h5py.File(pth, 'w') as dst:
            dst.create_dataset(
                'taxonomy_tree',
                data=taxonomy_tree_fixture.to_str().encode('utf-8'))
            dst.create_dataset(
                'cluster_to_row',
                data=json.dumps(cluster_to_row).encode('utf-8'))
            dst.create_dataset(
                'n_cells',
                data=n_cells)

    return result


def test_leaf_census(
        expected_census,
        taxonomy_tree_fixture):
    precompute_path_list = list(expected_census['a'].keys())
    (actual_census,
     actual_tree) = run_leaf_census(precompute_path_list)
    assert actual_census == expected_census
    assert actual_tree.is_equal_to(taxonomy_tree_fixture)


def test_merge_precompute(
        tmp_dir_fixture):

    rng = np.random.default_rng(771231)

    n_clusters = 6
    n_genes = 22

    pth0 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='AAA',
        suffix='.h5')
    n0 = np.array([10, 5, 7, 9, 13, 12])
    s0 = rng.random((n_clusters, n_genes))
    ssq0 = rng.random((n_clusters, n_genes))

    pth1 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='BBB',
        suffix='.h5')
    n1 = np.array([0, 0, 1000, 0, 0, 0])
    s1 = rng.random((n_clusters, n_genes))
    ssq1 = rng.random((n_clusters, n_genes))

    pth2 = mkstemp_clean(
       dir=tmp_dir_fixture,
       prefix='CCC',
       suffix='.h5')
    n2 = np.array([4, 11, 200, 6, 27, 8])
    s2 = rng.random((n_clusters, n_genes))
    ssq2 = rng.random((n_clusters, n_genes))

    cluster_to_row = {
        f'c{ii}':int(ii) for ii in range(n_clusters)
    }
    col_names = [f'g{ii}' for ii in range(n_genes)]

    tree = TaxonomyTree(
        data={
            'hierarchy': ['class', 'cluster'],
            'class': {
                'A': ['c0', 'c1'],
                'B': ['c2', 'c3', 'c4', 'c5']
            },
            'cluster': {
                f'c{ii}': [] for ii in range(n_clusters)
            }
        })

    for (pth, ncells, sumarr, sumsqarr) in [
                (pth0, n0, s0, ssq0),
                (pth1, n1, s1, ssq1),
                (pth2, n2, s2, ssq2)]:
        with h5py.File(pth, 'w') as dst:
            dst.create_dataset(
                'taxonomy_tree',
                data=tree.to_str(drop_cells=True).encode('utf-8'))
            dst.create_dataset(
                'cluster_to_row',
                data=json.dumps(cluster_to_row).encode('utf-8'))
            dst.create_dataset(
                'col_names',
                data=json.dumps(cluster_to_row).encode('utf-8'))
            dst.create_dataset(
                'metadata', data='abcd'.encode('utf-8'))
            dst.create_dataset(
                'n_cells', data=ncells)
            dst.create_dataset(
                'sum', data=sumarr)
            dst.create_dataset(
                'sumsq', data=sumsqarr)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='merged_',
        suffix='.h5')

    merge_precompute_files(
        precompute_path_list=[pth1, pth2, pth0],
        output_path=output_path)

    actual_tree = TaxonomyTree.from_precomputed_stats(output_path)
    assert actual_tree.is_equal_to(tree)

    with h5py.File(output_path, 'r') as src:
        actual_n_cells = src['n_cells'][()]
        actual_sumarr = src['sum'][()]
        actual_sumsqarr = src['sumsq'][()]

    expected_n = [n0, n1, n2]
    expected_s = [s0, s1, s2]
    expected_ssq = [ssq0, ssq1, ssq2]

    # which datasets do which clusters come from
    expected_idx = [0, 2, 1, 0, 2, 0]

    assert actual_n_cells.shape == (n_clusters,)
    assert actual_sumarr.shape == (n_clusters, n_genes)
    assert actual_sumsqarr.shape == (n_clusters, n_genes)

    for i_cluster in range(n_clusters):
        idx = expected_idx[i_cluster]

        assert actual_n_cells[i_cluster] == expected_n[idx][i_cluster]

        np.testing.assert_allclose(
            actual_sumarr[i_cluster, :],
            expected_s[idx][i_cluster, :],
            atol=0.0, rtol=1.0e-6)

        np.testing.assert_allclose(
            actual_sumsqarr[i_cluster, :],
            expected_ssq[idx][i_cluster, :],
            atol=0.0, rtol=1.0e-6)
