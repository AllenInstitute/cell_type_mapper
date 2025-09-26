import h5py
import numpy as np
import pathlib
import tempfile

import cell_type_mapper.utils.utils as ctm_utils
import cell_type_mapper.utils.anndata_utils as anndata_utils
import cell_type_mapper.type_assignment.election as election
import cell_type_mapper.type_assignment.marker_cache_v2 as marker_cache


def test_full_hann_election(
        tree_fixture,
        cell_by_gene_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture):

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture,
            prefix='hann_election_'
        )
    )

    reference_genes = cell_by_gene_fixture['reference'].gene_identifiers
    query_genes = anndata_utils.read_df_from_h5ad(
        query_h5ad_fixture,
        df_name='var'
    ).index.values
    query_cells = anndata_utils.read_df_from_h5ad(
        query_h5ad_fixture,
        df_name='obs'
    ).index.values

    marker_lookup = {
        'None': [f'g{ii}' for ii in
                 (1, 2, 3, 5, 11, 15, 16, 17, 18, 19, 20)],
        'class/A': [f'g{ii}' for ii in np.arange(12, 24)],
        'subclass/b': [f'g{ii}' for ii in np.arange(19)],
        'subclass/c': [f'g{ii}' for ii in np.arange(1, 27, 2)]
    }

    marker_cache_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hann_election_marker_cache_',
        suffix='.h5'
    )

    marker_cache.create_marker_cache_from_specified_markers(
        marker_lookup=marker_lookup,
        reference_gene_names=reference_genes,
        query_gene_names=list(query_genes),
        output_cache_path=marker_cache_path,
        log=None,
        taxonomy_tree=tree_fixture,
        min_markers=1
    )

    rng = np.random.default_rng(712310)

    bootstrap_iteration = 71
    chunk_size = 60

    bootstrap_factor_lookup = {
        'None': 0.5,
        'class': 0.5,
        'subclass': 0.5
    }

    result_path_list = election.run_type_assignment_on_h5ad_cpu(
        query_h5ad_path=query_h5ad_fixture,
        precomputed_stats_path=precomputed_stats_fixture,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=tree_fixture,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=bootstrap_iteration,
        n_processors=3,
        chunk_size=chunk_size,
        rng=rng,
        n_assignments=None,
        normalization='log2CPM',
        tmp_dir=tmp_dir_fixture,
        max_gb=3,
        output_taxonomy_tree=None,
        results_output_path=tmp_dir,
        algorithm='hann'
    )

    assert len(result_path_list) == 4
    for ii, pth in enumerate(result_path_list):
        with h5py.File(pth, 'r') as src:
            cell_identifiers = src['cell_identifiers'][()]
            votes = src['votes'][()]
            corr = src['correlation'][()]

        expected_cells = query_cells[ii*chunk_size:(ii+1)*chunk_size]
        np.testing.assert_array_equal(
            actual=[c.decode('utf-8') for c in cell_identifiers],
            desired=expected_cells
        )

        assert votes.shape == corr.shape
        assert votes.shape == (
            len(expected_cells),
            len(tree_fixture.nodes_at_level(tree_fixture.leaf_level))
        )

        # make sure each cell gets expected number of votes
        row_sum = votes.sum(axis=1)
        np.testing.assert_array_equal(
            row_sum,
            bootstrap_iteration*np.ones(len(expected_cells))
        )

        # make sure there is a diversity of vote values
        unq_votes = np.unique(votes)
        assert len(unq_votes) > 10
