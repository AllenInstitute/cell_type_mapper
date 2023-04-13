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


@pytest.fixture
def gt0_threshold():
    return 1


@pytest.mark.parametrize(
    "keep_all_stats, to_keep_frac, n_selection_processors",
    [
     (False, None, 4)
    ])
def test_marker_cache_pipeline(
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
    query_genes += ["nonsense_1", "nonsense_2", "nonsense_3"]
    rng.shuffle(query_genes)

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

    parent_node_list = [None]
    for level in taxonomy_tree['hierarchy'][:-1]:
        k_list = list(taxonomy_tree[level].keys())
        k_list.sort()
        for k in k_list:
            parent_node_list.append((level, k))

    ct = 0
    expected_query = set()
    with h5py.File(marker_cache_path, 'r') as actual_file:
        assert "all_query_genes" in actual_file.keys()
        for parent_node in parent_node_list:
            leaf_pair_list = get_all_leaf_pairs(
                taxonomy_tree=taxonomy_tree,
                parent_node=parent_node)

            if len(leaf_pair_list) > 0:
                expected = select_marker_genes(
                    score_path=score_path,
                    leaf_pair_list= leaf_pair_list,
                    query_genes=query_genes,
                    genes_per_pair=genes_per_pair,
                    rows_at_a_time=1000000,
                    n_processors=n_selection_processors)
            else:
                expected['reference'] = np.zeros(0, dtype=int)
                expected['query'] = np.zeros(0, dtype=int)

            ct += len(expected['reference'])

            if parent_node is None:
                actual_ref = actual_file['None']['reference'][()]
                actual_query = actual_file['None']['query'][()]
            else:
                grp = actual_file[parent_node[0]][parent_node[1]]
                actual_ref = grp['reference'][()]
                actual_query = grp['query'][()]

            expected_query = expected_query.union(set(expected['query']))

            np.testing.assert_array_equal(
                expected['reference'], actual_ref)
            np.testing.assert_array_equal(
                expected['query'], actual_query)

        # make sure that we correctly recorded all of the
        # marker genes needed from the query set
        expected_query = np.sort(np.array(list(expected_query)))
        assert len(expected_query) < len(query_genes)
        np.testing.assert_array_equal(
            actual_file['all_query_genes'][()],
            expected_query)

    # make sure we weren't testing all empty datasets
    assert ct > 0

    _clean_up(tmp_dir)
