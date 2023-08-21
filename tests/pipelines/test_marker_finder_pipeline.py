import pytest

import pandas as pd
import numpy as np
import h5py
import itertools
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
    find_markers_for_all_taxonomy_pairs,
    _find_markers_worker)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

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
        n_genes,
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
    n_pairs = len(siblings)

    find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=marker_path,
            n_processors=n_processors,
            tmp_dir=tmp_dir)

    with h5py.File(marker_path, 'r') as in_file:
        assert 'gene_names' in in_file
        assert json.loads(in_file['gene_names'][()].decode('utf-8')) == gene_names
        assert 'pair_to_idx' in in_file
        for sub_k in ("up_gene_idx", "up_pair_idx",
                      "down_gene_idx", "down_pair_idx"):
            assert f"sparse_by_pair/{sub_k}" in in_file
            assert f"sparse_by_gene/{sub_k}" in in_file

        pair_to_idx = json.loads(in_file['pair_to_idx'][()].decode('utf-8'))
        n_cols = in_file['n_pairs'][()]

    # check that we get the expected result
    precomputed_stats = read_precomputed_stats(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=True)

    tree_as_leaves = convert_tree_to_leaves(tree_fixture)

    marker_parent = MarkerGeneArray.from_cache_path(
        marker_path)

    assert gene_names == marker_parent.gene_names

    tot_markers = 0
    marker_sum = 0
    tot_up = 0
    up_sum = 0
    are_markers = set()

    # will compare by-gene access with these matrices
    # after we have looped through all pairs
    global_marker = np.zeros((n_genes, n_pairs), dtype=bool)
    global_up = np.zeros((n_genes, n_pairs), dtype=bool)

    for level in pair_to_idx:
        for node1 in pair_to_idx[level]:
            for node2 in pair_to_idx[level][node1]:
                cluster_stats = precomputed_stats['cluster_stats']
                idx = pair_to_idx[level][node1][node2]

                (_,
                 expected_markers,
                 expected_up_reg) = score_differential_genes(
                    node_1=f'{level}/{node1}',
                    node_2=f'{level}/{node2}',
                    precomputed_stats=cluster_stats,
                    p_th=0.01,
                    q1_th=0.5,
                    qdiff_th=0.7)

                # we won't have up=True unless
                # a gene is also a marker
                expected_up_reg[np.logical_not(expected_markers)] = False

                global_marker[:, idx] = expected_markers
                global_up[:, idx] = expected_up_reg

                for flag, name in zip(expected_markers, gene_names):
                    if flag:
                        are_markers.add(name)

                (actual_markers,
                 actual_up_reg) = marker_parent.marker_mask_from_pair_idx(idx)

                if expected_markers.sum() > 0:
                    assert actual_markers.sum() > 0

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

    assert len(are_markers) < len(gene_names)
    assert global_up.sum() > 0
    assert global_marker.sum() > 0

    for i_gene in range(n_genes):
        (actual_markers,
         actual_up) = marker_parent.marker_mask_from_gene_idx(i_gene)
        expected_markers = global_marker[i_gene, :]
        expected_up = global_up[i_gene, :]
        np.testing.assert_array_equal(
            actual_markers,
            expected_markers)
        np.testing.assert_array_equal(
            actual_up,
            expected_up)


@pytest.mark.parametrize(
    "p_th, q1_th, qdiff_th",
    itertools.product(
        (0.01, 0.02),
        (0.5, 0.4),
        (0.7, 0.5, 0.9)
    ))
def test_find_markers_worker(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_dir_fixture,
        n_genes,
        tree_fixture,
        p_th,
        q1_th,
        qdiff_th):

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
    siblings = get_all_pairs(tree_fixture)
    tree_as_leaves = taxonomy_tree.as_leaves

    precomputed_stats = read_precomputed_stats(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=True)

    idx_to_pair = dict()
    pair_to_idx = dict()
    siblings = taxonomy_tree.siblings
    n_pairs = len(siblings)
    for idx, sibling_pair in enumerate(siblings):
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]
        idx_to_pair[idx] = sibling_pair

        if level not in pair_to_idx:
            pair_to_idx[level] = dict()
        if node1 not in pair_to_idx[level]:
            pair_to_idx[level][node1] = dict()
        if node2 not in pair_to_idx[level]:
            pair_to_idx[level][node2] = dict()

        pair_to_idx[level][node1][node2] = idx

    _find_markers_worker(
        cluster_stats=precomputed_stats['cluster_stats'],
        tree_as_leaves=tree_as_leaves,
        idx_to_pair=idx_to_pair,
        n_genes=n_genes,
        p_th=p_th,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        tmp_path=marker_path)

    # check that we get the expected result

    with h5py.File(marker_path, 'r') as src:
        up_by_pair = scipy_sparse.csr_array(
            (np.ones(src['up_gene_idx'].shape[0], dtype=bool),
             src['up_gene_idx'][()],
             src['up_pair_idx'][()]),
            shape=(n_pairs, n_genes))

        down_by_pair = scipy_sparse.csr_array(
            (np.ones(src['down_gene_idx'].shape[0], dtype=bool),
             src['down_gene_idx'][()],
             src['down_pair_idx'][()]),
            shape=(n_pairs, n_genes))

    tot_markers = 0
    marker_sum = 0
    tot_up = 0
    up_sum = 0
    are_markers = set()
    for level in pair_to_idx:
        for node1 in pair_to_idx[level]:
            for node2 in pair_to_idx[level][node1]:
                cluster_stats = precomputed_stats['cluster_stats']

                (expected_score,
                 expected_markers,
                 expected_up_reg) = score_differential_genes(
                    node_1=f'{level}/{node1}',
                    node_2=f'{level}/{node2}',
                    precomputed_stats=cluster_stats,
                    p_th=0.01,
                    q1_th=0.5,
                    qdiff_th=0.7)

                idx = pair_to_idx[level][node1][node2]

                actual_markers = np.logical_or(
                    down_by_pair[[idx], :].toarray(),
                    up_by_pair[[idx], :].toarray())[0, :]

                np.testing.assert_array_equal(
                    expected_markers,
                    actual_markers)

                actual_up_reg = up_by_pair[[idx], :].toarray()[0, :]

                # we won't have up=True unless
                # a gene is also a marker
                expected_up_reg[np.logical_not(expected_markers)] = False

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
