import pytest
import numpy as np
import warnings

import cell_type_mapper.utils.distance_utils as distance_utils
import cell_type_mapper.taxonomy.taxonomy_tree as tree_module
import cell_type_mapper.cell_by_gene.cell_by_gene as cbg_module
import cell_type_mapper.hann_mapping.hann_election as hann_election


@pytest.fixture
def tree_fixture():
    tree_data = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {'A': ['a', 'b'], 'B': ['c']},
        'subclass': {'a': ['a1'],
                     'b': ['b1', 'b2'],
                     'c': ['c1', 'c2']},
        'cluster': {
            'a1': [],
            'b1': [],
            'b2': [],
            'c1': [],
            'c2': []}
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        taxonomy_tree = tree_module.TaxonomyTree(data=tree_data)
    return taxonomy_tree


@pytest.fixture
def cell_by_gene_fixture(tree_fixture):

    n_cells = 20
    n_clusters = len(tree_fixture.nodes_at_level(tree_fixture.leaf_level))
    n_genes = 30
    gene_identifiers = [f'g{ii}' for ii in range(n_genes)]

    rng = np.random.default_rng(881231)

    reference_data = np.zeros((n_clusters, n_genes), dtype=float)
    reference_data[0, 10:25] = np.sin(np.arange(15)*2.0*np.pi/7.0)
    reference_data[1, 0:18] = np.cos(np.arange(18)*2.0*np.pi/12.0)
    reference_data[2, 17:n_genes] = 2.0*(rng.random(13)-0.5)
    reference_data[3, 10:] = 1.2
    reference_data[4, :15] = 0.7
    reference_data[4, 15:] = np.linspace(0, 1.5, 15)

    reference = cbg_module.CellByGeneMatrix(
        data=reference_data,
        gene_identifiers=gene_identifiers,
        cell_identifiers=tree_fixture.nodes_at_level(
            tree_fixture.leaf_level),
        normalization='log2CPM'
    )

    query_data = np.zeros((n_cells, n_genes), dtype=float)
    for i_cells in range(n_cells):
        pair = rng.choice(np.arange(n_clusters), 2, replace=False)
        weights = rng.random(2)
        this = (weights[0]*reference_data[pair[0], :]
                + weights[1]*reference_data[pair[1], :]) / weights.sum()
        query_data[i_cells, :] = this

    query = cbg_module.CellByGeneMatrix(
        data=query_data,
        gene_identifiers=gene_identifiers,
        normalization='log2CPM'
    )

    return {'reference': reference, 'query': query}


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

    hann_election._assign_children_of_one_parent(
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

    hann_election._assign_children_of_one_parent(
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

    hann_election._assign_children_of_one_parent(
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

    hann_election._update_hann_votes(
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

    hann_election._update_hann_votes(
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
