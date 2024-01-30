import pytest

import copy
import h5py
import itertools
import json
import numpy as np
import scipy.sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

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

from cell_type_mapper.diff_exp.p_value_markers import (
    _find_markers_from_p_mask_worker,
    _get_validity_mask,
    find_markers_for_all_taxonomy_pairs_from_p_mask)


@pytest.fixture(scope='module')
def tmp_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('p_mask_markers_')
    yield tmp_dir
    _clean_up(tmp_dir)


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


@pytest.mark.parametrize(
    "n_valid, use_valid_gene_idx, q_min",
    itertools.product(
       (5, 30),
       (True, False),
       (0.0, 0.1)
    )
)
def test_p_mask_marker_worker(
        tmp_dir_fixture,
        precomputed_path_fixture,
        taxonomy_tree_fixture,
        cluster_pair_fixture,
        n_valid,
        use_valid_gene_idx,
        q_min):

    raw_valid_gene_idx = np.array(
                       [7, 11, 14, 21, 26, 31, 32, 34, 85,
                        86, 90, 92, 94, 104, 106, 107, 110,
                        111, 112, 115, 118, 119, 120, 122,
                        123, 125, 126, 183, 186, 191, 193,
                        195, 209, 211, 213, 214, 217, 228,
                        230, 232, 250, 254, 256, 260, 261,
                        262, 265, 270, 272])

    if use_valid_gene_idx:
        valid_gene_idx = raw_valid_gene_idx
    else:
        valid_gene_idx = None

    cluster_stats = read_precomputed_stats(
        precomputed_stats_path=precomputed_path_fixture,
        taxonomy_tree=taxonomy_tree_fixture,
        for_marker_selection=True)

    gene_names = cluster_stats['gene_names']
    cluster_stats = cluster_stats['cluster_stats']

    q1_th = 0.5
    qdiff_th = 0.7
    log2_fold_th = 1.0

    # these need to be so low because of how the test
    # data is constructed. These low thresholds give a difference
    # between n_valid = 10 and n_valid = 30
    p_th = 0.01
    q1_min_th = q_min
    qdiff_min_th = q_min
    log2_fold_min_th = 0.01

    p_mask_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='p_mask_for_markers_',
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
        n_processors=2)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='p_mask_worker_',
        suffix='.h5')

    with h5py.File(p_mask_path, 'r') as src:
        pair_to_idx = json.loads(
            src['pair_to_idx'][()].decode('utf-8'))

    raw_idx_to_pair = dict()
    for level in pair_to_idx:
        for cl0 in pair_to_idx[level]:
            for cl1 in pair_to_idx[level][cl0]:
                idx = pair_to_idx[level][cl0][cl1]
                raw_idx_to_pair[idx] = (level, cl0, cl1)

    pair0 = 16
    pair1 = 32

    raw_idx_values = list(raw_idx_to_pair.keys())
    raw_idx_values.sort()
    n_pairs_full = len(raw_idx_values)
    assert len(raw_idx_values) > 48
    these_idx_values = raw_idx_values[pair0:pair1]
    idx_to_pair = {
        idx: raw_idx_to_pair[idx]
        for idx in these_idx_values
    }

    _find_markers_from_p_mask_worker(
        p_value_mask_path=p_mask_path,
        cluster_stats=cluster_stats,
        tree_as_leaves=taxonomy_tree_fixture.as_leaves,
        idx_to_pair=idx_to_pair,
        n_genes=len(gene_names),
        tmp_path=output_path,
        n_valid=n_valid,
        valid_gene_idx=valid_gene_idx)

    with h5py.File(p_mask_path, 'r') as src:
        expected_distances = scipy.sparse.csr_array(
            (src['data'][()].astype(np.float64),
             src['indices'][()].astype(np.int64),
             src['indptr'][()].astype(np.int64)),
            shape=(n_pairs_full, len(gene_names)))

    expected_distances = expected_distances.toarray()

    # load in the result of finding the markers for this chunk
    with h5py.File(output_path, 'r') as src:
        assert src['up_gene_idx'].shape[0] > 0
        assert src['down_gene_idx'].shape[0] > 0

        up_indptr = src['up_pair_idx'][()].astype(np.int64)
        up_indices = src['up_gene_idx'][()].astype(np.int64)
        up_data = np.ones(up_indices.shape, dtype=int)
        up_array = scipy.sparse.csr_array(
                (up_data, up_indices, up_indptr),
                shape=(pair1-pair0, len(gene_names)))
        up_array = up_array.toarray()

        down_indptr = src['down_pair_idx'][()].astype(np.int64)
        down_indices = src['down_gene_idx'][()].astype(np.int64)
        down_data = np.ones(down_indices.shape, dtype=int)
        down_array = scipy.sparse.csr_array(
                (down_data, down_indices, down_indptr),
                shape=(pair1-pair0, len(gene_names)))
        down_array = down_array.toarray()

    assert up_array.sum() > 0
    assert down_array.sum() > 0

    # construct the expected up- and down-regulated markers
    # by brute force
    expected_up = np.zeros(up_array.shape, dtype=int)
    expected_down = np.zeros(down_array.shape, dtype=int)
    eps = 1.0e-6

    for i_pair in range(pair0, pair1, 1):
        gene_indices = np.where(np.abs(expected_distances[i_pair, :]) >= eps)[0]
        raw_distances = expected_distances[i_pair, gene_indices]
        validity = _get_validity_mask(
            n_valid=n_valid,
            n_genes=len(gene_names),
            gene_indices=gene_indices,
            raw_distances=raw_distances,
            valid_gene_idx=valid_gene_idx)

        this_pair = raw_idx_to_pair[i_pair]
        node0 = f'{this_pair[0]}/{this_pair[1]}'
        node1 = f'{this_pair[0]}/{this_pair[2]}'
        mean0 = cluster_stats[node0]['mean']
        mean1 = cluster_stats[node1]['mean']
        mean_mask = (mean0 < mean1)

        up_mask = np.logical_and(
            validity, mean_mask)
        expected_up[i_pair-pair0, up_mask] = 1
        down_mask = np.logical_and(
            validity,
            np.logical_not(mean_mask))
        expected_down[i_pair-pair0, down_mask] = 1

    np.testing.assert_array_equal(
            up_array, expected_up)
    np.testing.assert_array_equal(
            down_array, expected_down)

    # make sure that valid_gene_idx made a difference
    used_genes = down_array.sum(axis=0) + up_array.sum(axis=0)
    assert used_genes.shape == (len(gene_names),)
    used_genes = np.where(used_genes>0)[0]
    if use_valid_gene_idx:
        assert set(used_genes) == set(raw_valid_gene_idx)
    else:
        assert len(set(used_genes)) > len(set(raw_valid_gene_idx))



@pytest.mark.parametrize(
    "n_valid, use_valid_gene_idx, q_min, n_processors, drop_level",
    itertools.product(
       (5, 30),
       (True, False),
       (0.0, 0.1),
       (1, 3),
       (None, 'subclass')
    )
)
def test_p_mask_marker_smoke(
        tmp_dir_fixture,
        precomputed_path_fixture,
        taxonomy_tree_fixture,
        cluster_pair_fixture,
        n_valid,
        use_valid_gene_idx,
        q_min,
        n_processors,
        drop_level):
    """
    smoke test for marker selection from p-value mask
    """
    n_pairs = len(cluster_pair_fixture)
    raw_valid_gene_idx = np.array(
                       [7, 11, 14, 21, 26, 31, 32, 34, 85,
                        86, 90, 92, 94, 104, 106, 107, 110,
                        111, 112, 115, 118, 119, 120, 122,
                        123, 125, 126, 183, 186, 191, 193,
                        195, 209, 211, 213, 214, 217, 228,
                        230, 232, 250, 254, 256, 260, 261,
                        262, 265, 270, 272])


    cluster_stats = read_precomputed_stats(
        precomputed_stats_path=precomputed_path_fixture,
        taxonomy_tree=taxonomy_tree_fixture,
        for_marker_selection=True)

    gene_names = cluster_stats['gene_names']
    cluster_stats = cluster_stats['cluster_stats']

    if use_valid_gene_idx:
        gene_list = [
            gene_names[ii]
            for ii in raw_valid_gene_idx]
    else:
        gene_list = None

    q1_th = 0.5
    qdiff_th = 0.7
    log2_fold_th = 1.0

    # these need to be so low because of how the test
    # data is constructed. These low thresholds give a difference
    # between n_valid = 10 and n_valid = 30
    p_th = 0.01
    q1_min_th = q_min
    qdiff_min_th = q_min
    log2_fold_min_th = 0.01

    p_mask_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='p_mask_for_markers_',
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
        n_processors=2)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='p_mask_pipeline_',
        suffix='.h5')

    find_markers_for_all_taxonomy_pairs_from_p_mask(
        precomputed_stats_path=precomputed_path_fixture,
        p_value_mask_path=p_mask_path,
        output_path=output_path,
        n_processors=n_processors,
        tmp_dir=tmp_dir_fixture,
        max_gb=10,
        n_valid=n_valid,
        gene_list=gene_list,
        drop_level=drop_level)

    # check that some markers were found
    with h5py.File(output_path, 'r') as src:
        for direction in ('up', 'down'):
            for axis in ('pair', 'gene'):
                if axis == 'pair':
                    other = 'gene'
                    sparse_class = scipy.sparse.csr_array
                else:
                    other = 'pair'
                    sparse_class = scipy.sparse.csc_array
                grp = src[f'sparse_by_{axis}']
                indices = grp[f'{direction}_{other}_idx'][()]
                indptr = grp[f'{direction}_{axis}_idx'][()]
                data = np.ones(indices.shape)
                arr = sparse_class(
                    (data,
                     indices.astype(int),
                     indptr.astype(int)),
                    shape=(n_pairs, len(gene_names))).toarray()
                assert arr.sum() > 0.0
