import h5py
import numpy as np

import cell_type_mapper.utils.utils as ctm_utils
import cell_type_mapper.utils.distance_utils as distance_utils
import cell_type_mapper.type_assignment.marker_cache_v2 as marker_cache
import cell_type_mapper.hann_mapping.hann_mapping as hann_mapping
import cell_type_mapper.type_assignment.election as election


def test_hann_children_of_one_parent(
        tree_fixture,
        cell_by_gene_fixture):
    """
    Test single HANN assignment from a specified parent
    """
    reference = cell_by_gene_fixture['reference']
    query = cell_by_gene_fixture['query']

    marker_lookup = {
        'class/A': np.arange(7, 25)
    }

    gene_subset_idx = np.array(
        [7, 9, 14, 15, 16, 19, 20, 22, 23]
    )

    query_data = np.copy(query.data)
    query_data = query_data[:, gene_subset_idx]
    reference_data = np.copy(reference.data)
    reference_data = reference_data[:, gene_subset_idx]
    reference_data = reference_data[np.array([0, 1, 2]), :]

    expected_nn = distance_utils.correlation_nearest_neighbors(
        baseline_array=reference_data,
        query_array=query_data,
        return_correlation=False
    )

    class dummy_rng(object):
        def choice(self, arr, n, replace=False):
            return gene_subset_idx

    cell_assignments = np.array(['']*query.n_cells)
    new_cell_assignments = np.array(['']*query.n_cells)
    correlation_vector = np.zeros(query.n_cells)

    active_cells = np.arange(0, query.n_cells, 2)
    expected_nn = expected_nn[active_cells]
    expected_assn = np.array([
        {0: 'a', 1: 'b', 2: 'b'}[nn] for nn in expected_nn
    ])

    cell_assignments[active_cells] = 'A'
    cell_assignments[1] = 'B'
    cell_assignments[5] = 'C'

    hann_mapping._assign_children_of_one_parent(
        cell_assignments=cell_assignments,
        new_cell_assignments=new_cell_assignments,
        correlation_vector=correlation_vector,
        taxonomy_tree=tree_fixture,
        parent_level=tree_fixture.hierarchy[0],
        parent='A',
        marker_lookup=marker_lookup,
        rng=dummy_rng(),
        bootstrap_factor=0.5,
        min_chosen_markers=0,
        query_cell_by_gene=query,
        reference_cell_by_gene=reference
    )

    expected = np.array(['']*query.n_cells)
    for ii in range(query.n_cells):
        if ii % 2 == 0:
            expected[ii] = expected_assn[ii//2]

    np.testing.assert_array_equal(
        actual=new_cell_assignments,
        desired=expected
    )

    # make sure correlation vector was untouched
    np.testing.assert_allclose(
        actual=correlation_vector,
        desired=np.zeros(len(new_cell_assignments), dtype=float),
        atol=0.0,
        rtol=1.0e-6
    )


def test_hann_children_of_one_parent_from_root(
        tree_fixture,
        cell_by_gene_fixture):
    """
    Test single HANN assignment from the root of the
    taxonomy
    """
    reference = cell_by_gene_fixture['reference']
    query = cell_by_gene_fixture['query']

    marker_lookup = {
        'None': np.arange(7, 25)
    }

    gene_subset_idx = np.array(
        [7, 9, 14, 15, 16, 19, 20, 22, 23]
    )

    query_data = np.copy(query.data)
    query_data = query_data[:, gene_subset_idx]
    reference_data = np.copy(reference.data)
    reference_data = reference_data[:, gene_subset_idx]

    expected_nn = distance_utils.correlation_nearest_neighbors(
        baseline_array=reference_data,
        query_array=query_data,
        return_correlation=False
    )

    class dummy_rng(object):
        def choice(self, arr, n, replace=False):
            return gene_subset_idx

    cell_assignments = np.array(['']*query.n_cells)
    new_cell_assignments = np.array(['']*query.n_cells)
    correlation_vector = np.zeros(query.n_cells)

    expected_assn = np.array([
        {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B'}[nn]
        for nn in expected_nn
    ])

    hann_mapping._assign_children_of_one_parent(
        cell_assignments=cell_assignments,
        new_cell_assignments=new_cell_assignments,
        correlation_vector=correlation_vector,
        taxonomy_tree=tree_fixture,
        parent_level=None,
        parent=None,
        marker_lookup=marker_lookup,
        rng=dummy_rng(),
        bootstrap_factor=0.5,
        min_chosen_markers=0,
        query_cell_by_gene=query,
        reference_cell_by_gene=reference
    )

    np.testing.assert_array_equal(
        actual=new_cell_assignments,
        desired=expected_assn
    )

    # make sure correlation vector was untouched
    np.testing.assert_allclose(
        actual=correlation_vector,
        desired=np.zeros(len(new_cell_assignments), dtype=float),
        atol=0.0,
        rtol=1.0e-6
    )


def test_hann_children_of_one_leaf_parent(
        tree_fixture,
        cell_by_gene_fixture):
    """
    Test single HANN assignment from a specified parent
    just above leaf level of the taxonomy
    """
    reference = cell_by_gene_fixture['reference']
    query = cell_by_gene_fixture['query']

    marker_lookup = {
        'subclass/b': np.arange(7, 25)
    }

    gene_subset_idx = np.array(
        [7, 9, 14, 15, 16, 19, 20, 22, 23]
    )

    query_data = np.copy(query.data)
    query_data = query_data[:, gene_subset_idx]
    reference_data = np.copy(reference.data)
    reference_data = reference_data[:, gene_subset_idx]
    reference_data = reference_data[np.array([1, 2]), :]

    (expected_nn,
     expected_corr) = distance_utils.correlation_nearest_neighbors(
        baseline_array=reference_data,
        query_array=query_data,
        return_correlation=True
    )

    class dummy_rng(object):
        def choice(self, arr, n, replace=False):
            return gene_subset_idx

    cell_assignments = np.array(['']*query.n_cells)
    new_cell_assignments = np.array(['']*query.n_cells)
    correlation_vector = np.zeros(query.n_cells)

    active_cells = np.arange(0, query.n_cells, 2)
    expected_nn = expected_nn[active_cells]
    expected_assn = np.array([
        {0: 'b1', 1: 'b2'}[nn] for nn in expected_nn
    ])

    cell_assignments[active_cells] = 'b'
    cell_assignments[1] = 'a'
    cell_assignments[5] = 'c'

    hann_mapping._assign_children_of_one_parent(
        cell_assignments=cell_assignments,
        new_cell_assignments=new_cell_assignments,
        correlation_vector=correlation_vector,
        taxonomy_tree=tree_fixture,
        parent_level=tree_fixture.hierarchy[1],
        parent='b',
        marker_lookup=marker_lookup,
        rng=dummy_rng(),
        bootstrap_factor=0.5,
        min_chosen_markers=0,
        query_cell_by_gene=query,
        reference_cell_by_gene=reference
    )

    expected = np.array(['']*query.n_cells)
    for ii in range(query.n_cells):
        if ii % 2 == 0:
            expected[ii] = expected_assn[ii//2]

    np.testing.assert_array_equal(
        actual=new_cell_assignments,
        desired=expected
    )

    final_corr = np.zeros(query.n_cells, dtype=float)
    final_corr[active_cells] = expected_corr[active_cells]

    # make sure correlation vector was untouched
    np.testing.assert_allclose(
        actual=correlation_vector,
        desired=final_corr,
        atol=0.0,
        rtol=1.0e-6
    )

    # make sure we found something interesting
    assert len(np.unique(new_cell_assignments)) == 2


def test_update_hann_votes(
        tree_fixture,
        cell_by_gene_fixture):
    """
    test _update_hann_votes
    """

    reference = cell_by_gene_fixture['reference']
    n_cells = 3

    votes = np.zeros(
        (n_cells, reference.n_cells), dtype=int
    )
    corr = np.zeros(
        (n_cells, reference.n_cells), dtype=float
    )

    cell_assignments = ['a1', 'c2', 'c2']
    correlation_vector = [0.2, 0.4, 0.3]

    hann_mapping._update_hann_votes(
        cell_assignments=cell_assignments,
        correlation_vector=correlation_vector,
        reference_cell_by_gene=reference,
        votes_out=votes,
        corr_out=corr
    )

    expected_votes = np.zeros((n_cells, reference.n_cells), dtype=int)
    expected_votes[0, 0] = 1
    expected_votes[1, 4] = 1
    expected_votes[2, 4] = 1

    expected_corr = np.zeros((n_cells, reference.n_cells), dtype=float)
    expected_corr[0, 0] = 0.2
    expected_corr[1, 4] = 0.4
    expected_corr[2, 4] = 0.3

    np.testing.assert_array_equal(
        actual=votes,
        desired=expected_votes)

    np.testing.assert_allclose(
        actual=corr,
        desired=expected_corr,
        atol=0.0,
        rtol=1.0e-6
    )

    cell_assignments = ['b2', 'c1', 'c2']
    correlation_vector = [0.1, 0.5, 0.9]

    hann_mapping._update_hann_votes(
        cell_assignments=cell_assignments,
        correlation_vector=correlation_vector,
        reference_cell_by_gene=reference,
        votes_out=votes,
        corr_out=corr
    )

    expected_votes[0, 2] = 1
    expected_votes[1, 3] = 1
    expected_votes[2, 4] = 2

    expected_corr[0, 2] = 0.1
    expected_corr[1, 3] = 0.5
    expected_corr[2, 4] = 1.2

    np.testing.assert_array_equal(
        actual=votes,
        desired=expected_votes)

    np.testing.assert_allclose(
        actual=corr,
        desired=expected_corr,
        atol=0.0,
        rtol=1.0e-6
    )


def test_hann_iteration_smoke(
        tree_fixture,
        cell_by_gene_fixture):
    """
    Run smoketest on _hann_iteration
    """
    marker_lookup = {
        'None': np.array([1, 2, 3, 5, 11, 15, 16, 17, 18, 19, 20]),
        'class/A': np.arange(12, 24),
        'subclass/b': np.arange(19),
        'subclass/c': np.arange(1, 27, 2)
    }

    bootstrap_factor_lookup = {
        'None': 0.5,
        'class': 0.5,
        'subclass': 0.5
    }

    rng = np.random.default_rng(22131)
    reference = cell_by_gene_fixture['reference']
    query = cell_by_gene_fixture['query']

    votes = np.zeros((query.n_cells, reference.n_cells), dtype=int)
    corr = np.zeros((query.n_cells, reference.n_cells), dtype=float)

    n_iter = 5
    for ii in range(n_iter):
        hann_mapping._hann_iteration(
            query_cell_by_gene=query,
            reference_cell_by_gene=reference,
            taxonomy_tree=tree_fixture,
            marker_lookup=marker_lookup,
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            rng=rng,
            votes_out=votes,
            corr_out=corr,
            min_chosen_markers=5
        )
        assert votes.sum() == (ii+1)*query.n_cells

    # make sure there is a diversity of vote counts
    np.testing.assert_array_equal(
        actual=np.unique(votes),
        desired=np.arange(n_iter+1, dtype=int)
    )

    col_sum = votes.sum(axis=0)
    assert col_sum.shape == (reference.n_cells, )
    assert col_sum.min() > 0

    # make sure that corr was updated
    assert corr.sum() > 0.0
    # cannot do detailed check on corr > 0 where
    # votes > 0 because correlation could be negative


def test_hann_tally_votes_smoke(
        tmp_dir_fixture,
        tree_fixture,
        cell_by_gene_fixture):
    """
    Run smoketest of hann_tally_votes
    """
    query = cell_by_gene_fixture['query']
    reference = cell_by_gene_fixture['reference']
    marker_lookup = {
        'None': [f'g{ii}' for ii in
                 (1, 2, 3, 5, 11, 15, 16, 17, 18, 19, 20)],
        'class/A': [f'g{ii}' for ii in np.arange(12, 24)],
        'subclass/b': [f'g{ii}' for ii in np.arange(19)],
        'subclass/c': [f'g{ii}' for ii in np.arange(1, 27, 2)]
    }
    marker_cache_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hann_marker_cache_',
        suffix='.h5'
    )
    marker_cache.create_marker_cache_from_specified_markers(
        marker_lookup=marker_lookup,
        reference_gene_names=reference.gene_identifiers,
        query_gene_names=query.gene_identifiers,
        output_cache_path=marker_cache_path,
        log=None,
        taxonomy_tree=tree_fixture,
        min_markers=1
    )

    rng = np.random.default_rng(611991)
    bootstrap_iteration = 56

    bootstrap_factor_lookup = {
        'None': 0.5,
        'class': 0.5,
        'subclass': 0.5
    }

    result = hann_mapping.hann_tally_votes(
        full_query_data=query,
        leaf_node_matrix=reference,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=tree_fixture,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng
    )

    votes = result['votes']
    assert votes.sum() == bootstrap_iteration*query.n_cells
    # make sure there is a diversity of vote counts
    assert len(np.unique(votes)) > 5

    col_sum = votes.sum(axis=0)
    assert col_sum.shape == (reference.n_cells, )
    assert col_sum.min() > 0

    np.testing.assert_array_equal(
        actual=np.array(result['cluster_identifiers']),
        desired=np.array(reference.cell_identifiers)
    )

    np.testing.assert_array_equal(
        actual=np.array(result['cell_identifiers']),
        desired=np.array(query.cell_identifiers)
    )


def test_hann_mapping_chunk_smoke(
        tmp_dir_fixture,
        tree_fixture,
        cell_by_gene_fixture):
    """
    Run smoketest of election._run_type_assignment_on_h5ad_worker
    with algorithm == 'hann'
    """
    query = cell_by_gene_fixture['query']
    reference = cell_by_gene_fixture['reference']
    marker_lookup = {
        'None': [f'g{ii}' for ii in
                 (1, 2, 3, 5, 11, 15, 16, 17, 18, 19, 20)],
        'class/A': [f'g{ii}' for ii in np.arange(12, 24)],
        'subclass/b': [f'g{ii}' for ii in np.arange(19)],
        'subclass/c': [f'g{ii}' for ii in np.arange(1, 27, 2)]
    }
    marker_cache_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hann_marker_cache_',
        suffix='.h5'
    )
    marker_cache.create_marker_cache_from_specified_markers(
        marker_lookup=marker_lookup,
        reference_gene_names=reference.gene_identifiers,
        query_gene_names=query.gene_identifiers,
        output_cache_path=marker_cache_path,
        log=None,
        taxonomy_tree=tree_fixture,
        min_markers=1
    )

    rng = np.random.default_rng(611991)
    bootstrap_iteration = 56

    bootstrap_factor_lookup = {
        'None': 0.5,
        'class': 0.5,
        'subclass': 0.5
    }

    dst_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hann_chunk_',
        suffix='.h5'
    )

    election._run_type_assignment_on_h5ad_worker(
        query_cell_chunk=query,
        leaf_node_matrix=reference,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=tree_fixture,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng,
        n_assignments=None,
        results_output_path=dst_path,
        output_taxonomy_tree=None,
        algorithm="hann"
    )

    with h5py.File(dst_path, "r") as src:
        votes = src["votes"][()]
        corr = src["correlation"][()]
        cell_id = src["cell_identifiers"][()]
        cluster_id = src["cluster_identifiers"][()]

    assert votes.sum() == bootstrap_iteration*query.n_cells
    # make sure there is a diversity of vote counts
    assert len(np.unique(votes)) > 5

    col_sum = votes.sum(axis=0)
    assert col_sum.shape == (reference.n_cells, )
    assert col_sum.min() > 0

    assert corr.shape == votes.shape

    np.testing.assert_array_equal(
        desired=query.cell_identifiers,
        actual=[c.decode('utf-8') for c in cell_id]
    )

    np.testing.assert_array_equal(
        desired=reference.cell_identifiers,
        actual=[c.decode('utf-8') for c in cluster_id]
    )


def test_collate_hann_mappings(
        tmp_dir_fixture):

    rng = np.random.default_rng(21311)
    n_cells = 150
    n_clusters = 7
    expected_cell_id = np.array(
        [f'c{ii}'.encode('utf-8') for ii in range(n_cells)]
    )
    expected_votes = rng.integers(0, 256, (n_cells, n_clusters))
    expected_corr = rng.random((n_cells, n_clusters))
    expected_cluster_id = np.array(
        [f'cl_{ii}'.encode('utf-8') for ii in range(n_clusters)]
    )

    path_list = [
        ctm_utils.mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'
        )
        for ii in range(3)
    ]
    with h5py.File(path_list[0], 'w') as dst:
        dst.create_dataset(
            'cell_identifiers',
            data=expected_cell_id[:54]
        )
        dst.create_dataset(
            'votes',
            data=expected_votes[:54, :]
        )
        dst.create_dataset(
            'correlation',
            data=expected_corr[:54, :]
        )
        dst.create_dataset(
            'cluster_identifiers',
            data=expected_cluster_id
        )

    with h5py.File(path_list[1], 'w') as dst:
        dst.create_dataset(
            'cell_identifiers',
            data=expected_cell_id[54:91]
        )
        dst.create_dataset(
            'votes',
            data=expected_votes[54:91, :]
        )
        dst.create_dataset(
            'correlation',
            data=expected_corr[54:91, :]
        )
        dst.create_dataset(
            'cluster_identifiers',
            data=expected_cluster_id
        )

    with h5py.File(path_list[2], 'w') as dst:
        dst.create_dataset(
            'cell_identifiers',
            data=expected_cell_id[91:]
        )
        dst.create_dataset(
            'votes',
            data=expected_votes[91:, :]
        )
        dst.create_dataset(
            'correlation',
            data=expected_corr[91:, :]
        )
        dst.create_dataset(
            'cluster_identifiers',
            data=expected_cluster_id
        )

    dst_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='collated_hann_mapping_',
        suffix='.h5'
    )
    hann_mapping.collate_hann_mappings(
        tmp_path_list=path_list,
        dst_path=dst_path
    )

    with h5py.File(dst_path, 'r') as src:
        cell_id = src['cell_identifiers'][()]
        cluster_id = src['cluster_identifiers'][()]
        votes = src['votes'][()]
        correlation = src['correlation'][()]

    np.testing.assert_array_equal(
        actual=cell_id,
        desired=expected_cell_id
    )

    np.testing.assert_array_equal(
        actual=cluster_id,
        desired=expected_cluster_id
    )

    np.testing.assert_array_equal(
        actual=votes,
        desired=expected_votes
    )

    np.testing.assert_allclose(
        actual=correlation,
        desired=expected_corr,
        atol=0.0,
        rtol=1.0e-6
    )
