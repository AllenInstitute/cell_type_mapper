import pytest

import anndata
import copy
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

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.taxonomy.utils import (
    get_taxonomy_tree,
    _get_rows_from_tree,
    get_all_pairs,
    get_all_leaf_pairs)

from hierarchical_mapping.diff_exp.scores import (
    diffexp_score)

from hierarchical_mapping.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers)

from hierarchical_mapping.type_assignment.matching import (
    get_leaf_means,
    assemble_query_data)

from hierarchical_mapping.type_assignment.election_v2 import (
    run_type_assignment_on_h5ad_v2)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)



@pytest.mark.parametrize('sparse_query', [True, False])
def test_running_h5ad_election(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory,
        gene_names,
        sparse_query):
    """
    Just a smoke test
    """
    rng = np.random.default_rng(2213122)

    n_genes = len(gene_names)
    genes_to_keep = None
    n_selection_processors = 4

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('pipeline_process'))
    hdf5_tmp = tmp_dir / 'hdf5'
    hdf5_tmp.mkdir()
    score_path = tmp_dir / 'score_results.h5'
    marker_cache_path = tmp_dir / 'marker_cache.h5'


    precompute_path = tmp_dir / 'precomputed.h5'
    assert not precompute_path.is_file()

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        taxonomy_tree=None,
        output_path=precompute_path,
        rows_at_a_time=10000,
        normalization="log2CPM")

    assert precompute_path.is_file()

    with h5py.File(precompute_path, 'r') as src:
        taxonomy_tree_dict = json.loads(
            src['taxonomy_tree'][()].decode('utf-8'))
        taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)

    assert not score_path.is_file()

    # make sure flush_every is not an integer
    # divisor of the number of sibling pairs
    flush_every = 11
    n_processors = 3

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        output_path=score_path,
        flush_every=flush_every,
        n_processors=n_processors,
        tmp_dir=tmp_dir)

    assert score_path.is_file()

    rng = np.random.default_rng(556623)
    query_genes = rng.choice(gene_names, n_genes//3, replace=False)
    query_genes = list(query_genes)

    query_genes += ["nonsense_0", "nonsense_1", "nonsense_2"]
    rng.shuffle(query_genes)

    n_processors = 3
    chunk_size = 21
    n_query_cells = 2*n_processors*chunk_size + 11
    if sparse_query:
        query_data = np.zeros(n_query_cells*len(query_genes), dtype=float)
        chosen_dex = rng.choice(np.arange(len(query_data)),
                                n_query_cells*len(query_genes)//3,
                                replace=False)
        query_data[chosen_dex] = rng.random(len(chosen_dex))
        query_data = query_data.reshape((n_query_cells, len(query_genes)))
        query_data = scipy_sparse.csr_matrix(query_data)
    else:
        query_data = rng.random((n_query_cells, len(query_genes)))

    assert not marker_cache_path.is_file()

    genes_per_pair = 7

    create_marker_cache_from_reference_markers(
        output_cache_path=marker_cache_path,
        input_cache_path=score_path,
        query_gene_names=query_genes,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=genes_per_pair,
        n_processors=n_selection_processors)

    assert marker_cache_path.is_file()

    query_cell_names = [f'q{ii}' for ii in range(n_query_cells)]
    query_h5ad_path = tmp_dir / 'query.h5ad'
    assert not query_h5ad_path.is_file()

    obs_data = [{'name': q, 'junk': 'nonsense'}
                for q in query_cell_names]
    obs = pd.DataFrame(obs_data)
    obs = obs.set_index('name')

    a_data = anndata.AnnData(X=query_data,
                             obs=obs,
                             dtype=float)
    a_data.write_h5ad(query_h5ad_path)

    assert query_h5ad_path.is_file()

    bootstrap_factor = 0.8
    bootstrap_iteration = 23

    result = run_type_assignment_on_h5ad_v2(
            query_h5ad_path=query_h5ad_path,
            precomputed_stats_path=precompute_path,
            marker_gene_cache_path=marker_cache_path,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor=bootstrap_factor,
            bootstrap_iteration=bootstrap_iteration,
            rng=rng,
            tmp_dir=tmp_dir)

    _clean_up(tmp_dir)
