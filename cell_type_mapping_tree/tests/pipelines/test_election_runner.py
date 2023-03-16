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
    get_all_leaf_pairs)

from hierarchical_mapping.diff_exp.scores import (
    diffexp_score,
    score_all_taxonomy_pairs,
    rank_genes)

from hierarchical_mapping.zarr_creation.zarr_from_h5ad import (
    contiguous_zarr_from_h5ad)

from hierarchical_mapping.diff_exp.precompute import (
    precompute_summary_stats_from_contiguous_zarr)

from hierarchical_mapping.marker_selection.utils import (
    select_marker_genes)

from hierarchical_mapping.type_assignment.marker_cache import (
    create_marker_gene_cache)

from hierarchical_mapping.type_assignment.matching import (
    get_leaf_means,
    assemble_query_data)

from hierarchical_mapping.type_assignment.election import (
    choose_node,
    run_type_assignment)


@pytest.fixture
def gt0_threshold():
    return 1


@pytest.mark.parametrize(
    "keep_all_stats, to_keep_frac, n_selection_processors",
    [
     (False, None, 4)
    ])
def test_running_single_election(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory,
        gene_names,
        gt0_threshold,
        keep_all_stats,
        to_keep_frac,
        n_selection_processors):
    """
    Just a smoke test
    """
    rng = np.random.default_rng(2213122)

    n_genes = len(gene_names)
    if to_keep_frac is not None:
        genes_to_keep = n_genes // to_keep_frac
        assert genes_to_keep > 0
        assert genes_to_keep < n_genes
    else:
        genes_to_keep = None

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('pipeline_process'))
    zarr_path = tmp_dir / 'zarr.zarr'
    hdf5_tmp = tmp_dir / 'hdf5'
    hdf5_tmp.mkdir()
    score_path = tmp_dir / 'score_results.h5'
    marker_cache_path = tmp_dir / 'marker_cache.h5'

    contiguous_zarr_from_h5ad(
        h5ad_path=h5ad_path_fixture,
        zarr_path=zarr_path,
        taxonomy_hierarchy=column_hierarchy,
        zarr_chunks=100000,
        n_processors=3)

    precompute_path = tmp_dir / 'precomputed.h5'
    assert not precompute_path.is_file()

    precompute_summary_stats_from_contiguous_zarr(
        zarr_path=zarr_path,
        output_path=precompute_path,
        rows_at_a_time=1000,
        n_processors=3)

    assert precompute_path.is_file()

    metadata = json.load(
            open(zarr_path / 'metadata.json', 'rb'))
    taxonomy_tree = metadata["taxonomy_tree"]

    assert not score_path.is_file()

    # make sure flush_every is not an integer
    # divisor of the number of sibling pairs
    flush_every = 11
    n_processors = 3

    score_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=score_path,
            gt1_threshold=0,
            gt0_threshold=gt0_threshold,
            flush_every=flush_every,
            n_processors=n_processors,
            keep_all_stats=keep_all_stats,
            genes_to_keep=genes_to_keep)

    assert score_path.is_file()

    rng = np.random.default_rng(556623)
    query_genes = rng.choice(gene_names, n_genes//3, replace=False)
    query_genes = list(query_genes)

    query_genes += ["nonsense_0", "nonsense_1", "nonsense_2"]
    rng.shuffle(query_genes)

    n_query_cells = 446
    query_data = rng.random((n_query_cells, len(query_genes)))

    assert not marker_cache_path.is_file()

    genes_per_pair = 7

    create_marker_gene_cache(
        cache_path=marker_cache_path,
        score_path=score_path,
        query_gene_names=query_genes,
        taxonomy_tree=taxonomy_tree,
        marker_genes_per_pair=genes_per_pair,
        n_processors=n_selection_processors)

    assert marker_cache_path.is_file()

    leaf_lookup = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precompute_path)

    for parent_node in (None, ("level2", "l2d")):
        data_for_election = assemble_query_data(
            full_query_data=query_data,
            mean_profile_lookup=leaf_lookup,
            taxonomy_tree=taxonomy_tree,
            marker_cache_path=marker_cache_path,
            parent_node=parent_node)

        result = choose_node(
            query_gene_data=data_for_election['query_data'],
            reference_gene_data=data_for_election['reference_data'],
            reference_types=data_for_election['reference_types'],
            bootstrap_factor=0.8,
            bootstrap_iteration=23,
            rng=rng)

        assert result.shape == (n_query_cells,)

    _clean_up(tmp_dir)


@pytest.mark.parametrize(
    "keep_all_stats, to_keep_frac, n_selection_processors",
    [
     (False, None, 4)
    ])
def test_running_full_election(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory,
        gene_names,
        gt0_threshold,
        keep_all_stats,
        to_keep_frac,
        n_selection_processors):
    """
    Just a smoke test
    """
    rng = np.random.default_rng(2213122)

    n_genes = len(gene_names)
    if to_keep_frac is not None:
        genes_to_keep = n_genes // to_keep_frac
        assert genes_to_keep > 0
        assert genes_to_keep < n_genes
    else:
        genes_to_keep = None

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('pipeline_process'))
    zarr_path = tmp_dir / 'zarr.zarr'
    hdf5_tmp = tmp_dir / 'hdf5'
    hdf5_tmp.mkdir()
    score_path = tmp_dir / 'score_results.h5'
    marker_cache_path = tmp_dir / 'marker_cache.h5'

    contiguous_zarr_from_h5ad(
        h5ad_path=h5ad_path_fixture,
        zarr_path=zarr_path,
        taxonomy_hierarchy=column_hierarchy,
        zarr_chunks=100000,
        n_processors=3)

    precompute_path = tmp_dir / 'precomputed.h5'
    assert not precompute_path.is_file()

    precompute_summary_stats_from_contiguous_zarr(
        zarr_path=zarr_path,
        output_path=precompute_path,
        rows_at_a_time=1000,
        n_processors=3)

    assert precompute_path.is_file()

    metadata = json.load(
            open(zarr_path / 'metadata.json', 'rb'))
    taxonomy_tree = metadata["taxonomy_tree"]

    assert not score_path.is_file()

    # make sure flush_every is not an integer
    # divisor of the number of sibling pairs
    flush_every = 11
    n_processors = 3

    score_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=score_path,
            gt1_threshold=0,
            gt0_threshold=gt0_threshold,
            flush_every=flush_every,
            n_processors=n_processors,
            keep_all_stats=keep_all_stats,
            genes_to_keep=genes_to_keep)

    assert score_path.is_file()

    rng = np.random.default_rng(556623)
    query_genes = rng.choice(gene_names, n_genes//3, replace=False)
    query_genes = list(query_genes)

    query_genes += ["nonsense_0", "nonsense_1", "nonsense_2"]
    rng.shuffle(query_genes)

    n_query_cells = 446
    query_data = rng.random((n_query_cells, len(query_genes)))

    assert not marker_cache_path.is_file()

    genes_per_pair = 7

    create_marker_gene_cache(
        cache_path=marker_cache_path,
        score_path=score_path,
        query_gene_names=query_genes,
        taxonomy_tree=taxonomy_tree,
        marker_genes_per_pair=genes_per_pair,
        n_processors=n_selection_processors)

    assert marker_cache_path.is_file()

    result = run_type_assignment(
        full_query_gene_data=query_data,
        precomputed_stats_path=precompute_path,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree,
        bootstrap_factor=0.8,
        bootstrap_iteration=23,
        rng=rng)

    assert len(result) == n_query_cells
    for i_cell in range(n_query_cells):
        for level in taxonomy_tree['hierarchy']:
            assert result[i_cell][level] is not None

    # check that every cell is assigned to a
    # taxonomically consistent set of types
    hierarchy = taxonomy_tree['hierarchy']
    for i_cell in range(n_query_cells):
        this_cell = result[i_cell]
        assert this_cell[hierarchy[0]] in taxonomy_tree[hierarchy[0]].keys()
        for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
            assert this_cell[child_level] in taxonomy_tree[parent_level][this_cell[parent_level]]

    _clean_up(tmp_dir)
