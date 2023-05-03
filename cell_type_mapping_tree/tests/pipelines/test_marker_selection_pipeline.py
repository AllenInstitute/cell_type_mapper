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

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.taxonomy.utils import (
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


@pytest.mark.parametrize(
    "keep_all_stats, to_keep_frac, from_root, n_selection_processors",
    [(True, None, False, 3),
     (False, None, False, 3),
     (False, 3, False, 3),
     (False, None, True, 3),
     (False, None, True, 1)
    ])
def test_marker_selection_pipeline(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory,
        gene_names,
        keep_all_stats,
        to_keep_frac,
        from_root,
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
    taxonomy_tree_dict = metadata["taxonomy_tree"]
    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)

    assert not score_path.is_file()

    # make sure flush_every is not an integer
    # divisor of the number of sibling pairs
    flush_every = 11
    n_processors = 3

    score_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=score_path,
            flush_every=flush_every,
            n_processors=n_processors,
            keep_all_stats=keep_all_stats,
            genes_to_keep=genes_to_keep)

    assert score_path.is_file()

    rng = np.random.default_rng(556623)
    query_genes = rng.choice(gene_names, n_genes//3, replace=False)
    query_genes = list(query_genes)

    if from_root:
        parent_node = None
    else:
        level = taxonomy_tree_dict['hierarchy'][1]
        k_list = list(taxonomy_tree_dict[level].keys())
        k_list.sort()
        parent_node = (level, k_list[0])

    leaf_pair_list = get_all_leaf_pairs(
            taxonomy_tree=taxonomy_tree_dict,
            parent_node=parent_node)

    marker_genes = select_marker_genes(
        score_path=score_path,
        leaf_pair_list=leaf_pair_list,
        query_genes=query_genes,
        genes_per_pair=5,
        rows_at_a_time=27,
        n_processors=n_selection_processors)

    assert len(marker_genes['reference']) > 0

    with h5py.File(score_path, 'r') as in_file:
        reference_gene_names = json.loads(
            in_file['gene_names'][()].decode('utf-8'))
    assert len(marker_genes['reference']) == len(marker_genes['query'])
    for ii in range(len(marker_genes['reference'])):
        rr = marker_genes['reference'][ii]
        qq = marker_genes['query'][ii]
        assert gene_names[rr] == query_genes[qq]

    _clean_up(tmp_dir)
