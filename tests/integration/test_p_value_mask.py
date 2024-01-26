import pytest

import copy
import h5py
import itertools
import json
import numpy as np
import scipy.sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.score_utils import (
    read_precomputed_stats,
    pij_from_stats,
    q_score_from_pij)

from cell_type_mapper.diff_exp.scores import (
    diffexp_p_values_from_stats,
    penetrance_parameter_distance)

from cell_type_mapper.diff_exp.p_value_mask import (
    create_p_value_mask_file)


@pytest.fixture(scope='module')
def taxonomy_tree_fixture(precomputed_path_fixture):
    return TaxonomyTree.from_precomputed_stats(
                precomputed_path_fixture)


@pytest.fixture(scope='module')
def cluster_pair_fixture(taxonomy_tree_fixture):
    leaf_list = copy.deepcopy(taxonomy_tree_fixture.all_leaves)
    leaf_list.sort()
    return list(itertools.combinations(leaf_list, 2))


def get_marker_stats(
        precomputed_path,
        cluster_pair_list,
        taxonomy_tree,
        p_th):

    stats = read_precomputed_stats(
        precomputed_stats_path=precomputed_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=True)

    results = dict()

    pvalue_array = np.zeros(
        (len(cluster_pair_list), len(stats['gene_names'])),
        dtype=float)
    fold_array = np.zeros(
        (len(cluster_pair_list), len(stats['gene_names'])),
        dtype=float)
    q1_array = np.zeros(
        (len(cluster_pair_list), len(stats['gene_names'])),
        dtype=float)
    qdiff_array = np.zeros(
        (len(cluster_pair_list), len(stats['gene_names'])),
        dtype=float)

    for i_pair, pair in enumerate(cluster_pair_list):
        node_1 = f'{taxonomy_tree.leaf_level}/{pair[0]}'
        node_2 = f'{taxonomy_tree.leaf_level}/{pair[1]}'

        pval = diffexp_p_values_from_stats(
            node_1=node_1,
            node_2=node_2,
            precomputed_stats=stats['cluster_stats'],
            p_th=p_th,
            big_nu=None,
            boring_t=None)

        pvalue_array[i_pair, :] = pval

        (pij_1,
         pij_2,
         log2_fold) = pij_from_stats(
             cluster_stats=stats['cluster_stats'],
             node_1=node_1,
             node_2=node_2)

        (q1,
         qdiff) = q_score_from_pij(
                     pij_1=pij_1,
                     pij_2=pij_2)

        fold_array[i_pair, :] = log2_fold
        q1_array[i_pair, :] = q1
        qdiff_array[i_pair, :] = qdiff


    results['pvalue'] = pvalue_array
    results['q1'] = q1_array
    results['qdiff'] = qdiff_array
    results['log2_fold'] = fold_array

    return results


# need q1, adiff, and log2_fold arrays
# then check that all combinations of passage exist

# first make sure that just using this test data
# has all the degeneracies I want to test for


@pytest.mark.parametrize(
    "q1_min_th, qdiff_min_th, log2_fold_min_th, p_th, n_processors",
    itertools.product(
        (0.0, 0.1),
        (0.0, 0.1),
        (0.0, 0.2, 0.8),
        (0.05, 0.01),
        (1, 3)
    ))
def test_dummy_p_value_mask(
        tmp_dir_fixture,
        precomputed_path_fixture,
        taxonomy_tree_fixture,
        cluster_pair_fixture,
        q1_min_th,
        qdiff_min_th,
        log2_fold_min_th,
        p_th,
        n_processors):

    marker_stats = get_marker_stats(
        precomputed_path=precomputed_path_fixture,
        taxonomy_tree=taxonomy_tree_fixture,
        cluster_pair_list=cluster_pair_fixture,
        p_th=p_th)

    q1_th = 0.5
    qdiff_th = 0.7
    log2_fold_th = 1.0

    # create expected distances

    stats = read_precomputed_stats(
        precomputed_stats_path=precomputed_path_fixture,
        taxonomy_tree=taxonomy_tree_fixture,
        for_marker_selection=True)

    distance_array = np.zeros(
        (len(cluster_pair_fixture), len(stats['gene_names'])),
        dtype=float)

    invalid_array = np.zeros(
        (len(cluster_pair_fixture), len(stats['gene_names'])),
        dtype=bool)

    for i_pair in range(len(cluster_pair_fixture)):
        dist = penetrance_parameter_distance(
            q1_score=marker_stats['q1'][i_pair, :],
            qdiff_score=marker_stats['qdiff'][i_pair, :],
            log2_fold=marker_stats['log2_fold'][i_pair, :],
            q1_th=q1_th,
            q1_min_th=q1_min_th,
            qdiff_th=qdiff_th,
            qdiff_min_th=qdiff_min_th,
            log2_fold_th=log2_fold_th,
            log2_fold_min_th=log2_fold_min_th)

        invalid_array[i_pair, :] = dist['invalid']
        wgt = dist['wgt']
        wgt[(wgt==0.0)] = -1.0
        distance_array[i_pair, :] = wgt

    p_mask_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='p_mask_',
        suffix='.h5')

    create_p_value_mask_file(
        precomputed_stats_path=precomputed_path_fixture,
        dst_path=p_mask_path,
        p_th=p_th,
        q1_th=q1_th,
        q1_min_th=q1_min_th,
        qdiff_th=qdiff_th,
        qdiff_min_th=qdiff_min_th,
        log2_fold_th=log2_fold_th,
        log2_fold_min_th=log2_fold_min_th,
        tmp_dir=tmp_dir_fixture,
        n_processors=n_processors)

    with h5py.File(p_mask_path, 'r') as src:
        gene_names = json.loads(src['gene_names'][()].decode('utf-8'))
        assert gene_names == stats['gene_names']
        pair_to_idx = json.loads(src['pair_to_idx'][()].decode('utf-8'))

        assert src['indices'].dtype == src['indptr'].dtype
        assert src['data'][()].min() < 0.0

        actual = scipy.sparse.csr_array(
            (src['data'][()].astype(float),
             src['indices'][()],
             src['indptr'][()]),
            shape=(len(cluster_pair_fixture), len(stats['gene_names'])))

    # check order of clusters
    level = taxonomy_tree_fixture.leaf_level

    for ii, pair in enumerate(cluster_pair_fixture):
        n0 = f'{pair[0]}'
        n1 = f'{pair[1]}'
        assert pair_to_idx[level][n0][n1] == ii

    actual = actual.toarray()

    expected_invalid = np.logical_or(
        np.logical_not(marker_stats['pvalue'] < p_th),
        invalid_array
        )

    distance_array[expected_invalid] = 0.0

    assert expected_invalid.sum() < distance_array.size
    assert actual.min() < 0.0

    np.testing.assert_allclose(
        actual,
        distance_array,
        rtol=0.0,
        atol=np.finfo(np.float16).resolution)
