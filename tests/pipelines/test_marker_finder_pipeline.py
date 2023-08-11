import pytest

import pandas as pd
import numpy as np
import h5py
import anndata
import pathlib
import json
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.taxonomy.utils import (
    get_taxonomy_tree,
    _get_rows_from_tree,
    get_all_pairs,
    convert_tree_to_leaves)

from cell_type_mapper.diff_exp.scores import (
    read_precomputed_stats,
    score_differential_genes)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.binary_array.binary_array import (
    BinarizedBooleanArray)

from cell_type_mapper.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.marker_selection.marker_array_purely_sparse import (
    MarkerGeneArrayPureSparse)

from cell_type_mapper.marker_selection.selection import (
    select_marker_genes_v2)


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
        tmp_dir_fixture,
        gene_names,
        tree_fixture):

    tmp_dir = tmp_dir_fixture

    marker_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            suffix='.h5'))

    precompute_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            suffix='.h5'))

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        taxonomy_tree=None,
        output_path=precompute_path,
        rows_at_a_time=1000)

    with h5py.File(precompute_path, 'r') as in_file:
        assert len(in_file['n_cells'][()]) > 0

    taxonomy_tree = TaxonomyTree(data=tree_fixture)

    n_processors = 3
    siblings = get_all_pairs(tree_fixture)

    find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=marker_path,
            n_processors=n_processors,
            tmp_dir=tmp_dir)

    with h5py.File(marker_path, 'r') as in_file:
        assert 'gene_names' in in_file
        assert 'full_gene_names' in in_file
        assert 'pair_to_idx' in in_file
        for sub_k in ("up_gene_idx", "up_pair_idx",
                      "down_gene_idx", "down_pair_idx"):
            assert f"sparse_by_pair/{sub_k}" in in_file
            assert f"sparse_by_gene/{sub_k}" in in_file

        pair_to_idx = json.loads(in_file['pair_to_idx'][()].decode('utf-8'))
        n_cols = in_file['n_pairs'][()]
        filtered_gene_names = set(json.loads(
                                      in_file['gene_names'][()].decode('utf-8')))

    # check that we get the expected result
    precomputed_stats = read_precomputed_stats(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=True)

    tree_as_leaves = convert_tree_to_leaves(tree_fixture)

    raw_gene_to_idx = {gene: ii for ii, gene in enumerate(gene_names)}

    marker_parent = MarkerGeneArrayPureSparse.from_cache_path(
        marker_path)

    filtered_idx_to_gene = {ii: gene
                            for ii, gene in enumerate(marker_parent.gene_names)}

    markers = BinarizedBooleanArray(
        n_rows=len(filtered_gene_names),
        n_cols=n_cols)
    up_regulated = BinarizedBooleanArray(
        n_rows=len(filtered_gene_names),
        n_cols=n_cols)

    for i_pair in range(n_cols):
        (marker_mask,
         up_mask) = marker_parent.marker_mask_from_pair_idx(
                        pair_idx=i_pair)
        markers.set_col(i_pair, marker_mask)
        up_regulated.set_col(i_pair, up_mask)

    tot_markers = 0
    marker_sum = 0
    tot_up = 0
    up_sum = 0
    are_markers = set()
    for level in pair_to_idx:
        for node1 in pair_to_idx[level]:
            for node2 in pair_to_idx[level][node1]:
                cluster_stats = precomputed_stats['cluster_stats']

                (_,
                 raw_expected_markers,
                 raw_expected_up_reg) = score_differential_genes(
                    node_1=f'{level}/{node1}',
                    node_2=f'{level}/{node2}',
                    precomputed_stats=cluster_stats,
                    p_th=0.01,
                    q1_th=0.5,
                    qdiff_th=0.7)

                for flag, name in zip(raw_expected_markers, gene_names):
                    if flag:
                        are_markers.add(name)

                expected_markers = []
                expected_up_reg = []
                for ig, gn in enumerate(gene_names):
                    if gn in filtered_gene_names:
                        expected_markers.append(raw_expected_markers[ig])
                        expected_up_reg.append(raw_expected_up_reg[ig])

                expected_markers = np.array(expected_markers)
                expected_up_reg = np.array(expected_up_reg)

                idx = pair_to_idx[level][node1][node2]
                actual_markers = markers.get_col(idx)
                actual_up_reg = up_regulated.get_col(idx)

                if expected_markers.sum() > 0:
                    assert actual_markers.sum() > 0
                np.testing.assert_array_equal(
                    expected_markers,
                    actual_markers)

                # we won't have up=True unless
                # a gene is also a marker
                expected_up_reg = expected_up_reg[expected_markers]
                actual_up_reg = actual_up_reg[expected_markers]

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

    # make sure that the marker file only kept genes that ever occur as markers
    assert len(are_markers) < len(gene_names)
    assert are_markers == set(filtered_gene_names)
