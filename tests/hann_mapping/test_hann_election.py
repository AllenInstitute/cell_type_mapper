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
        level=tree_fixture.hierarchy[0],
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
        new_cell_assignments,
        expected
    )
