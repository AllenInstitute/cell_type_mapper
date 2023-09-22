import pytest

import h5py
import itertools
import json
import numpy as np
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.diff_exp.scores import (
    read_precomputed_stats,
    score_differential_genes)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.cli.reference_markers import (
    ReferenceMarkerRunner)


def test_scoring_with_min_thresholds(
        taxonomy_tree_fixture,
        threshold_mask_fixture,
        precomputed_fixture,
        n_genes):
    """
    Test score_differential_genes with different thresholds on the
    penetrance statistics. Set n_valid so large that only the
    min thresholds will matter.
    """

    stats = read_precomputed_stats(
        precomputed_stats_path=precomputed_fixture,
        taxonomy_tree=taxonomy_tree_fixture,
        for_marker_selection=True)

    qdiff_min_th_list = [0.01, 0.4]
    log2_fold_min_th_list = [0.01, 0.5]

    cluster_name_list = list(taxonomy_tree_fixture.nodes_at_level('cluster'))

    for cl0, cl1 in itertools.combinations(cluster_name_list, 2):
        for test_case in threshold_mask_fixture[cl0][cl1]:
            config = test_case['config']

            (score,
             valid,
             up) = score_differential_genes(
                         node_1=f'cluster/{cl0}',
                         node_2=f'cluster/{cl1}',
                         precomputed_stats=stats['cluster_stats'],
                         p_th=config['p_th'],
                         q1_th=1.0,
                         q1_min_th=config['q1_min_th'],
                         qdiff_th=1.0,
                         qdiff_min_th=config['qdiff_min_th'],
                         log2_fold_th=1.0,
                         log2_fold_min_th=config['log2_fold_min_th'],
                         n_cells_min=2,
                         boring_t=None,
                         big_nu=None,
                         exact_penetrance=False,
                         n_valid=n_genes*10)
            
            np.testing.assert_array_equal(
                valid,
                test_case['expected'])


@pytest.mark.parametrize('use_cli', [True, False])
def test_find_all_markers_given_min_thresholds(
        threshold_mask_fixture_all_pairs,
        precomputed_fixture,
        taxonomy_tree_fixture,
        n_genes,
        tmp_dir_fixture,
        use_cli):
    """
    Test marker gene selection in case where penetrance thresholds
    are uninteresting, such that only the min thresholds (specified in
    threshold_mask_fixture_all_pairs) apply. This allows us to check
    for exact agreement between actual marker genes and expected marker
    genes.
    """

    for test_case in threshold_mask_fixture_all_pairs:
        config = test_case['config']
        h5_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                prefix='reference_marker_test_',
                suffix='.h5'))

        if use_cli:
            cli_config = {
                'precomputed_path': str(precomputed_fixture),
                'output_path': str(h5_path),
                'tmp_dir': str(tmp_dir_fixture),
                'n_processors': 3,
                'exact_penetrance': False,
                'p_th': config['p_th'],
                'q1_th': 1.0,
                'q1_min_th': config['q1_min_th'],
                'qdiff_th': 1.0,
                'qdiff_min_th': config['qdiff_min_th'],
                'log2_fold_th': 1.0,
                'log2_fold_min_th': config['log2_fold_min_th'],
                'n_valid': 1000,
                'clobber': True,
                'drop_level': None
            }

            runner = ReferenceMarkerRunner(
                args=[],
                input_data=cli_config)
            runner.run()

        else:
            find_markers_for_all_taxonomy_pairs(
                precomputed_stats_path=precomputed_fixture,
                taxonomy_tree=taxonomy_tree_fixture,
                output_path=h5_path,
                p_th=config['p_th'],
                q1_th=1.0,
                qdiff_th=1.0,
                log2_fold_th=1.0,
                q1_min_th=config['q1_min_th'],
                qdiff_min_th=config['qdiff_min_th'],
                log2_fold_min_th=config['log2_fold_min_th'],
                n_processors=3,
                tmp_dir=tmp_dir_fixture,
                max_gb=1,
                exact_penetrance=False,
                n_valid=1000)

        marker_array = MarkerGeneArray.from_cache_path(
            cache_path=h5_path)

        # assemble an (n_genes, n_pairs) array of validity indicators
        # so that we can test marker_array.marker_mask_from_gene_idx as well
        expected_array = np.zeros((n_genes, marker_array.n_pairs), dtype=bool)
        leaf_pairs = []

        for cl0 in test_case['expected']:
            for cl1 in test_case['expected'][cl0]:
                expected = test_case['expected'][cl0][cl1]
                pair_idx = marker_array.idx_of_pair(
                    level=taxonomy_tree_fixture.leaf_level,
                    node1=cl0,
                    node2=cl1)
                (actual,
                 _) = marker_array.marker_mask_from_pair_idx(pair_idx)
                np.testing.assert_array_equal(actual, expected)
                expected_array[:, pair_idx] = expected
                leaf_pairs.append(pair_idx)

        leaf_pairs = np.sort(np.array(leaf_pairs))

        for i_gene in range(n_genes):
            # This will be a bit more complicated because marker_array, for legacy
            # reasons, has pairs above the leaf level, which are not included
            # in our ground truth set.
            (actual,
            _) = marker_array.marker_mask_from_gene_idx(i_gene)
            np.testing.assert_array_equal(
                actual[leaf_pairs],
                expected_array[i_gene, leaf_pairs])

        if h5_path.exists():
            h5_path.unlink()


@pytest.mark.parametrize(
    "d_q1,d_qdiff,d_log2_fold,n_valid,use_cli",
    itertools.product(
        [0.1, 0.2, 100.0],
        [0.1, 0.2, 100.0],
        [0.1, 0.2, 100.0],
        [30, 20],
        [True, False]
    ))
def test_find_all_markers_given_thresholds(
        threshold_mask_fixture_all_pairs,
        precomputed_fixture,
        taxonomy_tree_fixture,
        n_genes,
        tmp_dir_fixture,
        d_q1,
        d_qdiff,
        d_log2_fold,
        n_valid,
        use_cli):
    """
    Have interesting values for q1_th, qdiff_th, and log2_fold_th.

    d_q1, d_qdiff, d_log2_fold are increments on top of the minimum
    thresholds specified in the test data fixture.

    Because ground truth was only calculated with uninteresting values
    for those thresholds, test can only assure that no genes that shouldn't
    be markers make it through.
    """

    for test_case in threshold_mask_fixture_all_pairs:
        config = test_case['config']
        h5_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                prefix='reference_marker_test_',
                suffix='.h5'))

        if use_cli:
            cli_config = {
                'precomputed_path': str(precomputed_fixture),
                'output_path': str(h5_path),
                'tmp_dir': str(tmp_dir_fixture),
                'n_processors': 3,
                'exact_penetrance': False,
                'p_th': config['p_th'],
                'q1_th': config['q1_min_th']+d_q1,
                'q1_min_th': config['q1_min_th'],
                'qdiff_th': config['qdiff_min_th']+d_qdiff,
                'qdiff_min_th': config['qdiff_min_th'],
                'log2_fold_th': config['log2_fold_min_th']+d_log2_fold,
                'log2_fold_min_th': config['log2_fold_min_th'],
                'n_valid': n_valid,
                'clobber': True,
                'drop_level': None
            }

            runner = ReferenceMarkerRunner(
                args=[],
                input_data=cli_config)
            runner.run()
        else:

            find_markers_for_all_taxonomy_pairs(
                precomputed_stats_path=precomputed_fixture,
                taxonomy_tree=taxonomy_tree_fixture,
                output_path=h5_path,
                p_th=config['p_th'],
                q1_th=config['q1_min_th']+d_q1,
                qdiff_th=config['qdiff_min_th']+d_qdiff,
                log2_fold_th=config['log2_fold_min_th']+d_log2_fold,
                q1_min_th=config['q1_min_th'],
                qdiff_min_th=config['qdiff_min_th'],
                log2_fold_min_th=config['log2_fold_min_th'],
                n_processors=3,
                tmp_dir=tmp_dir_fixture,
                max_gb=1,
                exact_penetrance=False,
                n_valid=n_valid)

        marker_array = MarkerGeneArray.from_cache_path(
            cache_path=h5_path)

        # assemble an (n_genes, n_pairs) array of validity indicators
        # so that we can test marker_array.marker_mask_from_gene_idx as well
        expected_array = np.zeros((n_genes, marker_array.n_pairs), dtype=bool)
        leaf_pairs = []

        for cl0 in test_case['expected']:
            for cl1 in test_case['expected'][cl0]:
                expected = test_case['expected'][cl0][cl1]
                pair_idx = marker_array.idx_of_pair(
                    level=taxonomy_tree_fixture.leaf_level,
                    node1=cl0,
                    node2=cl1)
                (actual,
                 _) = marker_array.marker_mask_from_pair_idx(pair_idx)
                expected_not = np.where(np.logical_not(expected))[0]
                np.testing.assert_array_equal(
                    actual[expected_not],
                    np.zeros(len(expected_not), dtype=bool))
                expected_array[:, pair_idx] = expected
                leaf_pairs.append(pair_idx)

        leaf_pairs = np.sort(np.array(leaf_pairs))

        for i_gene in range(n_genes):
            # This will be a bit more complicated because marker_array, for legacy
            # reasons, has pairs above the leaf level, which are not included
            # in our ground truth set.
            (actual,
            _) = marker_array.marker_mask_from_gene_idx(i_gene)

            actual = actual[leaf_pairs]
            expected = expected_array[i_gene, leaf_pairs]
            expected_not = np.where(np.logical_not(expected))[0]

            np.testing.assert_array_equal(
                actual[expected_not],
                np.zeros(len(expected_not), dtype=bool))

        if h5_path.exists():
            h5_path.unlink()
