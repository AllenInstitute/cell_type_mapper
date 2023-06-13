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

from hierarchical_mapping.type_assignment.election import (
    choose_node,
    run_type_assignment,
    run_type_assignment_on_h5ad)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


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
        taxonomy_tree = TaxonomyTree.from_str(
            src['taxonomy_tree'][()].decode('utf-8'))

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

    n_query_cells = 446
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

    with h5py.File(marker_cache_path, 'r') as in_file:
        query_gene_id = json.loads(
                             in_file["query_gene_names"][()].decode("utf-8"))
        query_markers = [query_gene_id[ii]
                         for ii in in_file['all_query_markers'][()]]

    query_cell_by_gene = CellByGeneMatrix(
        data=query_data,
        gene_identifiers=query_gene_id,
        normalization="log2CPM")

    query_cell_by_gene.downsample_genes_in_place(
        selected_genes=query_markers)

    leaf_matrix = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precompute_path)

    for parent_node in (None, ("level2", "l2d")):
        data_for_election = assemble_query_data(
            full_query_data=query_cell_by_gene,
            mean_profile_matrix=leaf_matrix,
            taxonomy_tree=taxonomy_tree,
            marker_cache_path=marker_cache_path,
            parent_node=parent_node)

        (result,
         confidence) = choose_node(
            query_gene_data=data_for_election['query_data'].data,
            reference_gene_data=data_for_election['reference_data'].data,
            reference_types=data_for_election['reference_types'],
            bootstrap_factor=0.8,
            bootstrap_iteration=23,
            rng=rng)

        assert result.shape == (n_query_cells,)
        assert confidence.shape == result.shape

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
    precompute_path = tmp_dir / 'precomputed.h5'

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

    n_query_cells = 446
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
    with h5py.File(marker_cache_path, 'r') as in_file:
        query_gene_id = json.loads(
                             in_file["query_gene_names"][()].decode("utf-8"))
        query_markers = [query_gene_id[ii]
                         for ii in in_file['all_query_markers'][()]]

    query_cell_by_gene = CellByGeneMatrix(
        data=query_data,
        gene_identifiers=query_gene_id,
        normalization="log2CPM")

    query_cell_by_gene.downsample_genes_in_place(
        selected_genes=query_markers)

    # get a CellByGeneMatrix of average expression
    # profiles for each leaf in the taxonomy
    leaf_node_matrix = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precompute_path)

    result = run_type_assignment(
        full_query_gene_data=query_cell_by_gene,
        leaf_node_matrix=leaf_node_matrix,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree,
        bootstrap_factor=0.8,
        bootstrap_iteration=23,
        rng=rng)

    assert len(result) == n_query_cells
    for i_cell in range(n_query_cells):
        for level in taxonomy_tree_dict['hierarchy']:
            assert result[i_cell][level] is not None

    # check that every cell is assigned to a
    # taxonomically consistent set of types
    hierarchy = taxonomy_tree_dict['hierarchy']
    for i_cell in range(n_query_cells):
        this_cell = result[i_cell]
        for level in hierarchy:
            assert level in this_cell
        for k in this_cell:
            assert this_cell[k] is not None
        assert this_cell[hierarchy[0]]['assignment'] in taxonomy_tree_dict[hierarchy[0]].keys()
        for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
            assert this_cell[child_level]['assignment'] in taxonomy_tree_dict[parent_level][this_cell[parent_level]['assignment']]

    _clean_up(tmp_dir)


@pytest.mark.parametrize(
    "keep_all_stats, to_keep_frac, n_selection_processors",
    [
     (False, None, 4)
    ])
def test_running_flat_election(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory,
        gene_names,
        keep_all_stats,
        to_keep_frac,
        n_selection_processors):
    """
    Just a smoke test in case where taxonomy has one level
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
        raw_tree = json.loads(src['taxonomy_tree'][()].decode('utf-8'))

    taxonomy_tree = dict()
    leaf_level = raw_tree['hierarchy'][-1]
    taxonomy_tree['hierarchy'] = [leaf_level]
    taxonomy_tree[leaf_level] = raw_tree[leaf_level]
    valid_types = set(taxonomy_tree[leaf_level].keys())

    taxonomy_tree_dict = copy.deepcopy(taxonomy_tree)
    taxonomy_tree = TaxonomyTree(data=taxonomy_tree)

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

    n_query_cells = 446
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

    with h5py.File(marker_cache_path, 'r') as in_file:
        query_gene_id = json.loads(
                             in_file["query_gene_names"][()].decode("utf-8"))
        query_markers = [query_gene_id[ii]
                         for ii in in_file['all_query_markers'][()]]

    query_cell_by_gene = CellByGeneMatrix(
        data=query_data,
        gene_identifiers=query_gene_id,
        normalization="log2CPM")

    query_cell_by_gene.downsample_genes_in_place(
        selected_genes=query_markers)

    # get a CellByGeneMatrix of average expression
    # profiles for each leaf in the taxonomy
    leaf_node_matrix = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precompute_path)

    result = run_type_assignment(
        full_query_gene_data=query_cell_by_gene,
        leaf_node_matrix=leaf_node_matrix,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree,
        bootstrap_factor=0.8,
        bootstrap_iteration=23,
        rng=rng)

    assert len(result) == n_query_cells
    for i_cell in range(n_query_cells):
        for level in taxonomy_tree_dict['hierarchy']:
            assert result[i_cell][level] is not None
            assert result[i_cell][level]['assignment'] in valid_types

    _clean_up(tmp_dir)



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

    result = run_type_assignment_on_h5ad(
            query_h5ad_path=query_h5ad_path,
            precomputed_stats_path=precompute_path,
            marker_gene_cache_path=marker_cache_path,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor=bootstrap_factor,
            bootstrap_iteration=bootstrap_iteration,
            rng=rng)

    assert len(result) == n_query_cells
    for i_cell in range(n_query_cells):
        for level in taxonomy_tree_dict['hierarchy']:
            assert result[i_cell][level] is not None

    # check that every cell is assigned to a
    # taxonomically consistent set of types
    hierarchy = taxonomy_tree_dict['hierarchy']
    name_set = set()
    for i_cell in range(n_query_cells):
        this_cell = result[i_cell]
        for level in hierarchy:
            assert level in this_cell
        for k in this_cell:
            assert this_cell[k] is not None
        name_set.add(this_cell['cell_id'])
        assert this_cell[hierarchy[0]]['assignment'] in taxonomy_tree_dict[hierarchy[0]].keys()
        for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
            assert this_cell[child_level]['assignment'] in taxonomy_tree_dict[parent_level][this_cell[parent_level]['assignment']]

    # make sure all cell_ids were transcribed
    assert len(name_set) == len(result)
    assert len(name_set) == len(query_cell_names)
    assert name_set == set(query_cell_names)

    _clean_up(tmp_dir)
