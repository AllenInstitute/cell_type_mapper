import pytest

import pandas as pd
import numpy as np
import h5py
import anndata
import pathlib
import json
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.taxonomy_utils import (
    get_taxonomy_tree,
    _get_rows_from_tree,
    get_all_pairs,
    convert_tree_to_leaves)

from hierarchical_mapping.diff_exp.scores import (
    read_precomputed_stats,
    score_differential_genes)

from hierarchical_mapping.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)


@pytest.fixture
def tree_fixture(
        records_fixture,
        column_hierarchy):
    return get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)



def test_marker_finding_pipeline(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory,
        gene_names,
        tree_fixture):

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('pipeline_process'))
    hdf5_tmp = tmp_dir / 'hdf5'
    hdf5_tmp.mkdir()
    marker_path = tmp_dir / 'marker_results.h5'

    precompute_path = tmp_dir / 'precomputed.h5'
    assert not precompute_path.is_file()

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        output_path=precompute_path,
        rows_at_a_time=1000)

    assert precompute_path.is_file()

    taxonomy_tree = tree_fixture

    # make sure flush_every is not an integer
    # divisor of the number of sibling pairs
    flush_every = 11
    n_processors = 3
    siblings = get_all_pairs(tree_fixture)
    assert len(siblings) > (n_processors*flush_every)
    assert len(siblings) % (n_processors*flush_every) != 0

    assert not marker_path.is_file()
    find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=marker_path,
            n_processors=n_processors,
            tmp_dir=tmp_dir)

    assert marker_path.is_file()
    with h5py.File(marker_path, 'r') as in_file:
        assert 'markers/data' in in_file
        assert 'up_regulated/data' in in_file
        assert 'gene_names' in in_file
        assert 'pair_to_idx' in in_file
        pair_to_idx = json.loads(in_file['pair_to_idx'][()].decode('utf-8'))
        n_cols = in_file['n_pairs'][()]

    # check that we get the expected result
    precomputed_stats = read_precomputed_stats(precompute_path)
    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)

    markers = BackedBinarizedBooleanArray(
        h5_path=marker_path,
        h5_group='markers',
        n_rows=len(gene_names),
        n_cols=n_cols)

    up_regulated = BackedBinarizedBooleanArray(
        h5_path=marker_path,
        h5_group='up_regulated',
        n_rows=len(gene_names),
        n_cols=n_cols)

    tot_markers = 0
    marker_sum = 0
    tot_up = 0
    up_sum = 0
    are_markers = set()
    for level in pair_to_idx:
        for node1 in pair_to_idx[level]:
            for node2 in pair_to_idx[level][node1]:
                pop1 = tree_as_leaves[level][node1]
                pop2 = tree_as_leaves[level][node2]
                cluster_stats = precomputed_stats['cluster_stats']

                (_,
                 expected_markers,
                 expected_up_reg) = score_differential_genes(
                    leaf_population_1=pop1,
                    leaf_population_2=pop2,
                    precomputed_stats=cluster_stats,
                    p_th=0.01,
                    q1_th=0.5,
                    qdiff_th=0.7)

                idx = pair_to_idx[level][node1][node2]
                actual_markers = markers.get_col(idx)
                actual_up_reg = up_regulated.get_col(idx)
                are_markers = are_markers.union(
                    set(list(np.where(expected_markers)[0])))

                np.testing.assert_array_equal(
                    expected_markers,
                    actual_markers)

                np.testing.assert_array_equal(
                    expected_up_reg,
                    actual_up_reg)

                tot_markers += len(expected_markers)
                marker_sum += expected_markers.sum()
                tot_up += len(expected_up_reg)
                up_sum += expected_up_reg.sum()

    # make sure that not all up_regulated/marker flags were trivial
    # (where "trivial" means all True or all False)
    assert tot_markers > 0
    assert tot_up > 0
    assert marker_sum > 0
    assert marker_sum < tot_markers
    assert up_sum > 0
    assert up_sum < tot_up
    assert are_markers == set([ii for ii in range(len(gene_names))])
    _clean_up(tmp_dir)
