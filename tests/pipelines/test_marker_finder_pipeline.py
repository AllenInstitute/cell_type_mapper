import pytest

import numpy as np
import h5py
import itertools
import pathlib
import tempfile
import json
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.taxonomy.utils import (
    get_taxonomy_tree,
    get_all_pairs)

from cell_type_mapper.diff_exp.score_utils import (
    read_precomputed_stats)

from cell_type_mapper.diff_exp.scores import (
    score_differential_genes)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs,
    _find_markers_worker)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.marker_selection.marker_array import (
    MarkerGeneArray)

from cell_type_mapper.cli.reference_markers import (
   ReferenceMarkerRunner)

from cell_type_mapper.cli.cli_log import CommandLog


@pytest.fixture
def tree_fixture(
        records_fixture,
        column_hierarchy):
    return get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)


def test_reference_marker_finding_cli(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_dir_fixture,
        tree_fixture):
    """
    Just a smoke test
    """

    precompute_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'))

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        taxonomy_tree=None,
        output_path=precompute_path,
        rows_at_a_time=1000)

    output_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    config = {
        'n_processors': 3,
        'precomputed_path_list': [str(precompute_path.resolve().absolute())],
        'output_dir': output_dir,
        'drop_level': None
    }

    runner = ReferenceMarkerRunner(args=[], input_data=config)
    runner.run()

    # check that logs are recorded in output files
    output_dir = pathlib.Path(output_dir)
    file_path_list = [n for n in output_dir.iterdir()]
    assert len(file_path_list) > 0
    for file_path in file_path_list:
        with h5py.File(file_path, "r") as src:
            metadata = json.loads(src['metadata'][()].decode('utf-8'))
        assert 'log' in metadata
        assert len(metadata['log']) > 0


@pytest.mark.parametrize(
        "limit_genes,use_log",
        itertools.product([True, False], [True, False]))
def test_marker_finding_pipeline(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_dir_fixture,
        gene_names,
        n_genes,
        tree_fixture,
        limit_genes,
        use_log):
    """
    if limit_genes, pass a gene_list thrugh to
    find_markers_for_all_taxonomy_pairs, limiting the list
    of genes that are valid markers
    """

    if use_log:
        log = CommandLog()
    else:
        log = None

    if limit_genes:
        rng = np.random.default_rng(22131)
        chosen_genes = list(rng.choice(
                gene_names, n_genes//2, replace=False))

        valid_gene_idx = np.array([
            ii for ii in range(len(gene_names))
            if gene_names[ii] in chosen_genes])
    else:
        chosen_genes = None
        valid_gene_idx = None

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
    siblings = [s for s in siblings if s[0] == taxonomy_tree.leaf_level]
    n_pairs = len(siblings)

    find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=marker_path,
            n_processors=n_processors,
            tmp_dir=tmp_dir,
            gene_list=chosen_genes,
            log=log)

    with h5py.File(marker_path, 'r') as in_file:
        assert 'gene_names' in in_file
        assert json.loads(
                in_file['gene_names'][()].decode('utf-8')) == gene_names
        assert 'pair_to_idx' in in_file
        for sub_k in ("up_gene_idx", "up_pair_idx",
                      "down_gene_idx", "down_pair_idx"):
            assert f"sparse_by_pair/{sub_k}" in in_file
            assert f"sparse_by_gene/{sub_k}" in in_file

        pair_to_idx = json.loads(in_file['pair_to_idx'][()].decode('utf-8'))

    # check that we get the expected result
    precomputed_stats = read_precomputed_stats(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=True)

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

    n_changes = 0

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
                    qdiff_th=0.7,
                    valid_gene_idx=valid_gene_idx)

                if limit_genes:
                    # check if limiting the genes made a difference
                    (_,
                     unlimited_markers,
                     _) = score_differential_genes(
                        node_1=f'{level}/{node1}',
                        node_2=f'{level}/{node2}',
                        precomputed_stats=cluster_stats,
                        p_th=0.01,
                        q1_th=0.5,
                        qdiff_th=0.7,
                        valid_gene_idx=None)
                    if not np.array_equal(unlimited_markers, expected_markers):
                        n_changes += 1

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

    if limit_genes:
        assert n_changes > 0

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
        tmp_path=marker_path,
        log2_fold_th=1.0,
        q1_min_th=0.1,
        qdiff_min_th=0.1,
        log2_fold_min_th=0.8,
        exact_penetrance=True)

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
                    p_th=p_th,
                    q1_th=q1_th,
                    qdiff_th=qdiff_th,
                    exact_penetrance=True)

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


def test_score_differential_genes_limited(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_dir_fixture,
        n_genes,
        tree_fixture):
    """
    Test that we can artificially limit the genes that
    are considered valid markers
    """

    p_th = 0.01
    q1_th = 0.5
    qdiff_th = 0.7
    q1_min_th = 0.0
    qdiff_min_th = 0.0
    log2_fold_min_th = 0.0
    n_valid = 10

    tmp_dir = tmp_dir_fixture

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

    precomputed_stats = read_precomputed_stats(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=True)

    idx_to_pair = dict()
    pair_to_idx = dict()
    siblings = taxonomy_tree.siblings
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

    # figure out which genes are markers in the case
    # where we are not limiting available genes
    baseline_lookup = dict()
    unlimited_marker_idx = set()
    for level in pair_to_idx:
        for node1 in pair_to_idx[level]:
            for node2 in pair_to_idx[level][node1]:
                cluster_stats = precomputed_stats['cluster_stats']

                (unlimited_score,
                 unlimited_markers,
                 unlimited_up_reg) = score_differential_genes(
                    node_1=f'{level}/{node1}',
                    node_2=f'{level}/{node2}',
                    precomputed_stats=cluster_stats,
                    p_th=p_th,
                    q1_th=q1_th,
                    qdiff_th=qdiff_th,
                    q1_min_th=q1_min_th,
                    qdiff_min_th=qdiff_min_th,
                    log2_fold_min_th=log2_fold_min_th,
                    n_valid=n_valid)

                unlimited_marker_idx = unlimited_marker_idx.union(
                    set(np.where(unlimited_markers)[0]))

                baseline_lookup[f'{level}/{node1}/{node2}'] = unlimited_markers

    assert len(unlimited_marker_idx) > 4
    rng = np.random.default_rng(2232)
    idx_list = list(unlimited_marker_idx)
    idx_list.sort()
    chosen_markers = rng.choice(idx_list, len(idx_list)//2, replace=False)
    chosen_marker_set = set(chosen_markers)

    # make sure that specifying a limited set of markers excludes
    # those markers, but does not add new markers
    limited_marker_idx = set()

    # cases where marker masks differ
    ct_diff = 0

    # cases where new markers made it in because of filter
    ct_more = 0

    ct_all = 0
    for level in pair_to_idx:
        for node1 in pair_to_idx[level]:
            for node2 in pair_to_idx[level][node1]:
                ct_all += 1
                cluster_stats = precomputed_stats['cluster_stats']

                (limited_score,
                 limited_markers,
                 limited_up_reg) = score_differential_genes(
                    node_1=f'{level}/{node1}',
                    node_2=f'{level}/{node2}',
                    precomputed_stats=cluster_stats,
                    p_th=p_th,
                    q1_th=q1_th,
                    qdiff_th=qdiff_th,
                    q1_min_th=q1_min_th,
                    qdiff_min_th=qdiff_min_th,
                    log2_fold_min_th=log2_fold_min_th,
                    valid_gene_idx=chosen_markers,
                    n_valid=n_valid)

                baseline = baseline_lookup[
                    f'{level}/{node1}/{node2}']

                if not np.array_equal(limited_markers, baseline):
                    ct_diff += 1

                these_markers = set(
                    np.where(limited_markers)[0])

                baseline_markers = set(
                    np.where(baseline)[0])

                if len(these_markers-baseline_markers) > 0:
                    ct_more += 1

                limited_marker_idx = limited_marker_idx.union(
                    these_markers)

                assert len(these_markers-chosen_marker_set) == 0

    assert len(limited_marker_idx) > 2
    assert limited_marker_idx != unlimited_marker_idx
    assert len(unlimited_marker_idx-limited_marker_idx) > 0
    assert ct_diff > 0
    assert ct_more > 0
    assert ct_more < ct_all
    assert ct_diff < ct_all
