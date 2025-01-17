import pytest

import anndata
import copy
import pandas as pd
import numpy as np
import h5py
import pathlib
import json
import scipy.sparse as scipy_sparse
import tempfile
import os
import warnings

from cell_type_mapper.utils.torch_utils import (
    is_torch_available)

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers)

from cell_type_mapper.type_assignment.matching import (
    get_leaf_means,
    assemble_query_data)

from cell_type_mapper.type_assignment.election import (
    tally_votes,
    reshape_type_assignment,
    run_type_assignment)

from cell_type_mapper.type_assignment.election import (
    run_type_assignment_on_h5ad_cpu)

from cell_type_mapper.type_assignment.election_runner import (
    run_type_assignment_on_h5ad)

if is_torch_available():
    from cell_type_mapper.gpu_utils.type_assignment.election import (
        run_type_assignment_on_h5ad_gpu)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with h5py.File(precompute_path, 'r') as src:
            taxonomy_tree = TaxonomyTree.from_str(
                src['taxonomy_tree'][()].decode('utf-8'))

    assert not score_path.is_file()

    n_processors = 3

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        output_path=score_path,
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

        (votes,
         corr_sum,
         reference_types) = tally_votes(
            query_gene_data=data_for_election['query_data'].data,
            reference_gene_data=data_for_election['reference_data'].data,
            reference_types=data_for_election['reference_types'],
            bootstrap_factor=0.8,
            bootstrap_iteration=23,
            rng=rng)

        (result,
         confidence,
         avg_corr,
         _) = reshape_type_assignment(
            votes=votes,
            corr_sum=corr_sum,
            reference_types=reference_types,
            n_assignments=10)

        assert result.shape == (n_query_cells,)
        assert confidence.shape == result.shape
        assert avg_corr.shape == result.shape

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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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

        n_processors = 3

        find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=score_path,
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

        bootstrap_factor = 0.8
        bootstrap_factor_lookup = {
            level: bootstrap_factor
            for level in taxonomy_tree.hierarchy}
        bootstrap_factor_lookup['None'] = bootstrap_factor

        result = run_type_assignment(
            full_query_gene_data=query_cell_by_gene,
            leaf_node_matrix=leaf_node_matrix,
            marker_gene_cache_path=marker_cache_path,
            taxonomy_tree=taxonomy_tree,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
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
            assert (
                this_cell[hierarchy[0]]['assignment']
                in taxonomy_tree_dict[hierarchy[0]].keys()
            )
            for parent_level, child_level in zip(hierarchy[:-1],
                                                 hierarchy[1:]):
                assert (
                    this_cell[child_level]['assignment']
                    in taxonomy_tree_dict[parent_level][
                        this_cell[parent_level]['assignment']]
                )

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

    n_processors = 3

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        output_path=score_path,
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

    bootstrap_factor = 0.8
    bootstrap_factor_lookup = {
        level: bootstrap_factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = bootstrap_factor

    result = run_type_assignment(
        full_query_gene_data=query_cell_by_gene,
        leaf_node_matrix=leaf_node_matrix,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=23,
        rng=rng)

    assert len(result) == n_query_cells
    for i_cell in range(n_query_cells):
        for level in taxonomy_tree_dict['hierarchy']:
            assert result[i_cell][level] is not None
            assert result[i_cell][level]['assignment'] in valid_types

    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def precompute_stats_path_fixture(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_dir_fixture):

    precompute_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        taxonomy_tree=None,
        output_path=precompute_path,
        rows_at_a_time=10000,
        normalization="log2CPM")
    return precompute_path


@pytest.fixture(scope='module')
def taxonomy_tree_fixture(precompute_stats_path_fixture):

    with h5py.File(precompute_stats_path_fixture, 'r') as src:
        taxonomy_tree_dict = json.loads(
            src['taxonomy_tree'][()].decode('utf-8'))
        taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)
    return taxonomy_tree, taxonomy_tree_dict


@pytest.fixture(scope='module')
def marker_score_fixture(
        tmp_dir_fixture,
        taxonomy_tree_fixture,
        precompute_stats_path_fixture):

    score_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_marker_scores_',
        suffix='.h5')

    n_processors = 3

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precompute_stats_path_fixture,
        taxonomy_tree=taxonomy_tree_fixture[0],
        output_path=score_path,
        n_processors=n_processors,
        tmp_dir=tmp_dir_fixture)

    return score_path


@pytest.fixture(scope='module')
def query_gene_fixture(gene_names):

    n_genes = len(gene_names)

    rng = np.random.default_rng(556623)
    query_genes = rng.choice(gene_names, n_genes//3, replace=False)
    query_genes = list(query_genes)

    query_genes += ["nonsense_0", "nonsense_1", "nonsense_2"]
    rng.shuffle(query_genes)

    return query_genes


@pytest.fixture(scope='function')
def query_data_fixture(
        query_gene_fixture,
        request):

    sparse_query = request.param
    rng = np.random.default_rng(76123)
    query_genes = query_gene_fixture

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

    return query_data


@pytest.fixture(scope='module')
def query_marker_cache_fixture(
        tmp_dir_fixture,
        marker_score_fixture,
        query_gene_fixture,
        taxonomy_tree_fixture):

    cache_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_marker_fixture_',
        suffix='.h5')

    n_selection_processors = 4
    genes_per_pair = 7

    create_marker_cache_from_reference_markers(
        output_cache_path=cache_path,
        input_cache_path=marker_score_fixture,
        query_gene_names=query_gene_fixture,
        taxonomy_tree=taxonomy_tree_fixture[0],
        n_per_utility=genes_per_pair,
        n_processors=n_selection_processors)

    return cache_path


@pytest.fixture(scope='function')
def query_h5ad_fixture(
        tmp_dir_fixture,
        query_data_fixture,
        query_gene_fixture):

    query_h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_h5ad_',
        suffix='.h5ad')

    query_data = query_data_fixture
    n_query_cells = query_data.shape[0]

    query_cell_names = [f'q{ii}' for ii in range(n_query_cells)]

    obs_data = [{'name': q, 'junk': 'nonsense'}
                for q in query_cell_names]
    obs = pd.DataFrame(obs_data)
    obs = obs.set_index('name')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(X=query_data,
                                 obs=obs,
                                 dtype=float)

    a_data.write_h5ad(query_h5ad_path)

    return query_h5ad_path


@pytest.mark.parametrize(
        'query_data_fixture',
        [True, False],
        indirect=['query_data_fixture'])
def test_running_h5ad_election(
        precompute_stats_path_fixture,
        taxonomy_tree_fixture,
        query_data_fixture,
        query_marker_cache_fixture,
        query_h5ad_fixture):
    """
    Just a smoke test
    """
    rng = np.random.default_rng(6712312)

    taxonomy_tree = taxonomy_tree_fixture[0]
    taxonomy_tree_dict = taxonomy_tree_fixture[1]

    n_processors = 3
    chunk_size = 21

    bootstrap_factor = 0.8
    bootstrap_iteration = 23

    bootstrap_factor_lookup = {
        level: bootstrap_factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = bootstrap_factor

    result = run_type_assignment_on_h5ad_cpu(
            query_h5ad_path=query_h5ad_fixture,
            precomputed_stats_path=precompute_stats_path_fixture,
            marker_gene_cache_path=query_marker_cache_fixture,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=rng)

    query_data = query_data_fixture
    n_query_cells = query_data.shape[0]

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
        assert (
            this_cell[hierarchy[0]]['assignment']
            in taxonomy_tree_dict[hierarchy[0]].keys()
        )
        for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
            assert (
                this_cell[child_level]['assignment']
                in taxonomy_tree_dict[
                    parent_level][this_cell[parent_level]['assignment']]
            )

    a_data = anndata.read_h5ad(query_h5ad_fixture, backed='r')
    query_cell_names = a_data.obs.index.values

    # make sure all cell_ids were transcribed
    assert len(name_set) == len(result)
    assert len(name_set) == len(query_cell_names)
    assert name_set == set(query_cell_names)


@pytest.mark.parametrize(
        'query_data_fixture',
        [True, False],
        indirect=['query_data_fixture'])
def test_running_h5ad_election_with_tmp_dir(
        precompute_stats_path_fixture,
        taxonomy_tree_fixture,
        query_data_fixture,
        query_marker_cache_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture):
    """
    Test running election with results stored in a temporary dir
    and then de-serialized
    """
    rng_seed = 6712312

    taxonomy_tree = taxonomy_tree_fixture[0]

    n_processors = 3
    chunk_size = 21

    bootstrap_factor = 0.8
    bootstrap_iteration = 23

    bootstrap_factor_lookup = {
        level: bootstrap_factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = bootstrap_factor

    baseline_result = run_type_assignment_on_h5ad_cpu(
            query_h5ad_path=query_h5ad_fixture,
            precomputed_stats_path=precompute_stats_path_fixture,
            marker_gene_cache_path=query_marker_cache_fixture,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=np.random.default_rng(rng_seed))

    baseline_result = {
        cell['cell_id']: cell for cell in baseline_result}

    tmp_result_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='result_buffer_')

    result = run_type_assignment_on_h5ad_cpu(
            query_h5ad_path=query_h5ad_fixture,
            precomputed_stats_path=precompute_stats_path_fixture,
            marker_gene_cache_path=query_marker_cache_fixture,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=np.random.default_rng(rng_seed),
            results_output_path=tmp_result_dir)

    result = {cell['cell_id']: cell for cell in result}

    assert set(result.keys()) == set(baseline_result.keys())
    for cell_id in baseline_result.keys():
        baseline = baseline_result[cell_id]
        test = result[cell_id]
        for level in baseline:
            if level == 'cell_id':
                continue
            assert baseline[level]['assignment'] == test[level]['assignment']
            np.testing.assert_allclose(
                (baseline[level]['bootstrapping_probability'],
                 baseline[level]['avg_correlation']),
                (test[level]['bootstrapping_probability'],
                 test[level]['avg_correlation']))


@pytest.mark.skipif(not is_torch_available(), reason='no torch')
@pytest.mark.parametrize(
        'query_data_fixture',
        [True, False],
        indirect=['query_data_fixture'])
def test_running_h5ad_election_gpu(
        precompute_stats_path_fixture,
        taxonomy_tree_fixture,
        query_data_fixture,
        query_marker_cache_fixture,
        query_h5ad_fixture):
    """
    Test self-consistency of GPU type assignments
    """
    env_var = 'AIBS_BKP_USE_TORCH'
    rng = np.random.default_rng(6712312)

    taxonomy_tree = taxonomy_tree_fixture[0]
    taxonomy_tree_dict = taxonomy_tree_fixture[1]

    n_processors = 3
    chunk_size = 21

    bootstrap_factor = 0.8
    bootstrap_iteration = 23

    bootstrap_factor_lookup = {
        level: bootstrap_factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = bootstrap_factor

    os.environ[env_var] = 'true'
    result = run_type_assignment_on_h5ad_gpu(
        query_h5ad_path=query_h5ad_fixture,
        precomputed_stats_path=precompute_stats_path_fixture,
        marker_gene_cache_path=query_marker_cache_fixture,
        taxonomy_tree=taxonomy_tree,
        n_processors=n_processors,
        chunk_size=chunk_size,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng,
        results_output_path=None)
    os.environ[env_var] = ''

    query_data = query_data_fixture
    n_query_cells = query_data.shape[0]

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
        assert (
            this_cell[hierarchy[0]]['assignment']
            in taxonomy_tree_dict[hierarchy[0]].keys()
        )
        for parent_level, child_level in zip(hierarchy[:-1], hierarchy[1:]):
            assert (
                this_cell[child_level]['assignment']
                in taxonomy_tree_dict[parent_level][
                    this_cell[parent_level]['assignment']]
            )

    a_data = anndata.read_h5ad(query_h5ad_fixture, backed='r')
    query_cell_names = a_data.obs.index.values

    # make sure all cell_ids were transcribed
    assert len(name_set) == len(result)
    assert len(name_set) == len(query_cell_names)
    assert name_set == set(query_cell_names)


@pytest.mark.skipif(not is_torch_available(), reason='no torch')
@pytest.mark.parametrize(
        'query_data_fixture',
        [True, False],
        indirect=['query_data_fixture'])
def test_running_h5ad_election_with_tmp_dir_gpu(
        precompute_stats_path_fixture,
        taxonomy_tree_fixture,
        query_data_fixture,
        query_marker_cache_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture):
    """
    Test running election with results stored in a temporary dir
    and then de-serialized
    """
    rng_seed = 6712312

    taxonomy_tree = taxonomy_tree_fixture[0]

    n_processors = 3
    chunk_size = 21

    bootstrap_factor = 0.8
    bootstrap_iteration = 23

    bootstrap_factor_lookup = {
        level: bootstrap_factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = bootstrap_factor

    env_var = 'AIBS_BKP_USE_TORCH'

    os.environ[env_var] = 'true'
    baseline_result = run_type_assignment_on_h5ad_gpu(
            query_h5ad_path=query_h5ad_fixture,
            precomputed_stats_path=precompute_stats_path_fixture,
            marker_gene_cache_path=query_marker_cache_fixture,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=np.random.default_rng(rng_seed))

    baseline_result = {
        cell['cell_id']: cell for cell in baseline_result}

    tmp_result_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture,
        prefix='result_buffer_')

    result = run_type_assignment_on_h5ad_gpu(
            query_h5ad_path=query_h5ad_fixture,
            precomputed_stats_path=precompute_stats_path_fixture,
            marker_gene_cache_path=query_marker_cache_fixture,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=np.random.default_rng(rng_seed),
            results_output_path=tmp_result_dir)
    os.environ[env_var] = ''

    result = {cell['cell_id']: cell for cell in result}

    assert set(result.keys()) == set(baseline_result.keys())
    for cell_id in baseline_result.keys():
        baseline = baseline_result[cell_id]
        test = result[cell_id]
        for level in baseline:
            if level == 'cell_id':
                continue
            assert baseline[level]['assignment'] == test[level]['assignment']
            np.testing.assert_allclose(
                (baseline[level]['bootstrapping_probability'],
                 baseline[level]['avg_correlation']),
                (test[level]['bootstrapping_probability'],
                 test[level]['avg_correlation']))


@pytest.fixture(scope='function')
def query_h5ad_fixture_negative(
        tmp_dir_fixture,
        query_data_fixture,
        query_gene_fixture):

    query_h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_h5ad_',
        suffix='.h5ad')

    if isinstance(query_data_fixture, np.ndarray):
        query_data = np.copy(query_data_fixture)
        data_median = np.median(query_data)
        query_data[query_data < data_median] -= data_median
    else:
        data_median = np.median(query_data_fixture.data)
        new_data = np.copy(query_data_fixture.data)
        new_data[new_data < data_median] -= data_median
        query_data = scipy_sparse.csr_matrix(
            (new_data,
             query_data_fixture.indices,
             query_data_fixture.indptr),
            shape=query_data_fixture.shape)

    n_query_cells = query_data.shape[0]

    query_cell_names = [f'q{ii}' for ii in range(n_query_cells)]

    obs_data = [{'name': q, 'junk': 'nonsense'}
                for q in query_cell_names]
    obs = pd.DataFrame(obs_data)
    obs = obs.set_index('name')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(X=query_data,
                                 obs=obs,
                                 dtype=float)

    a_data.write_h5ad(query_h5ad_path)

    return query_h5ad_path


@pytest.mark.parametrize(
        'query_data_fixture',
        [True, False],
        indirect=['query_data_fixture'])
def test_running_h5ad_election_negative_expression(
        precompute_stats_path_fixture,
        taxonomy_tree_fixture,
        query_data_fixture,
        query_marker_cache_fixture,
        query_h5ad_fixture_negative):
    """
    Test that an error is raised if normalization == 'raw'
    and the minimum expression value in the query set is negative
    """
    rng = np.random.default_rng(6712312)

    taxonomy_tree = taxonomy_tree_fixture[0]

    n_processors = 3
    chunk_size = 21

    bootstrap_factor = 0.8
    bootstrap_iteration = 23

    bootstrap_factor_lookup = {
        level: bootstrap_factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = bootstrap_factor

    with pytest.raises(RuntimeError, match="must be >= 0"):
        run_type_assignment_on_h5ad(
            query_h5ad_path=query_h5ad_fixture_negative,
            precomputed_stats_path=precompute_stats_path_fixture,
            marker_gene_cache_path=query_marker_cache_fixture,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=rng,
            normalization='raw')

    # make sure it runs if normalization is 'log2CPM'
    run_type_assignment_on_h5ad(
        query_h5ad_path=query_h5ad_fixture_negative,
        precomputed_stats_path=precompute_stats_path_fixture,
        marker_gene_cache_path=query_marker_cache_fixture,
        taxonomy_tree=taxonomy_tree,
        n_processors=n_processors,
        chunk_size=chunk_size,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng,
        normalization='log2CPM')


@pytest.mark.parametrize(
        'query_data_fixture',
        [True, ],
        indirect=['query_data_fixture'])
def test_running_h5ad_election_duplicate_cell_ids(
        precompute_stats_path_fixture,
        taxonomy_tree_fixture,
        query_data_fixture,
        query_marker_cache_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture):
    """
    Test that an error is raised if obs.index.values contains
    repeat entries
    """
    rng = np.random.default_rng(6712312)

    taxonomy_tree = taxonomy_tree_fixture[0]

    n_processors = 3
    chunk_size = 21

    bootstrap_factor = 0.8
    bootstrap_iteration = 23

    bootstrap_factor_lookup = {
        level: bootstrap_factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = bootstrap_factor

    query_h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='repeated_obs_idx_',
        suffix='.h5ad'
    )

    src = anndata.read_h5ad(query_h5ad_fixture, backed='r')
    n_cells = len(src.obs)
    new_obs_data = []
    for i_cell in range(n_cells):
        if i_cell == 3 or i_cell == 5:
            cell_id = 'dummy'
        else:
            cell_id = f'c_{i_cell}'
        new_obs_data.append({'cell_id': cell_id})
    new_obs = pd.DataFrame(new_obs_data).set_index('cell_id')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        dst = anndata.AnnData(
            obs=new_obs,
            X=src.X,
            var=src.var
        )
    dst.write_h5ad(query_h5ad_path)

    with pytest.raises(RuntimeError, match="obs.index.values are not unique"):
        run_type_assignment_on_h5ad(
            query_h5ad_path=query_h5ad_path,
            precomputed_stats_path=precompute_stats_path_fixture,
            marker_gene_cache_path=query_marker_cache_fixture,
            taxonomy_tree=taxonomy_tree,
            n_processors=n_processors,
            chunk_size=chunk_size,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            bootstrap_iteration=bootstrap_iteration,
            rng=rng,
            normalization='log2CPM')
