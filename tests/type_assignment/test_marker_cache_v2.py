import pytest

import copy
import numpy as np

from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    validate_marker_lookup)


@pytest.fixture
def taxonomy_tree_fixture():

    data = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {
            'A': ['b', 'd', 'c'],
            'B': ['a'],
            'C': ['e', 'f', 'g']
        },
        'subclass': {
            'a': ['1', '2'],
            'b': ['3'],
            'c': ['4', '5'],
            'd': ['6', '7'],
            'e': ['8'],
            'f': ['9', '10'],
            'g': ['11', '12']
        },
        'cluster': { str(ii): [] for ii in range(1, 13, 1)}
    }
    tree = TaxonomyTree(data=data)
    return tree


@pytest.fixture
def query_gene_fixture():
    return [f'q_{ii}' for ii in range(7)]


@pytest.fixture
def complete_lookup_fixture(
        query_gene_fixture,
        taxonomy_tree_fixture):

    rng = np.random.default_rng(612312)

    lookup = dict()
    for parent in taxonomy_tree_fixture.all_parents:
        if parent is None:
            parent_str = 'None'
        else:
            parent_str = f'{parent[0]}/{parent[1]}'
        lookup[parent_str] = list(
            rng.choice(query_gene_fixture, 3, replace=False))
    return lookup

def test_validate_marker_lookup(
        taxonomy_tree_fixture,
        query_gene_fixture,
        complete_lookup_fixture):
    """
    Validate that an error is raised if markers are not available for all
    non-trivial parents
    """

    # test that valid case passes
    validate_marker_lookup(
        marker_lookup=complete_lookup_fixture,
        taxonomy_tree=taxonomy_tree_fixture,
        query_gene_names=query_gene_fixture)

    # test that it is okay to miss trivial parents
    lookup = copy.deepcopy(complete_lookup_fixture)
    lookup.pop('class/B')
    lookup['subclass/e'] = []
    lookup['subclass/b'] = ['x', 'y', 'z']
    validate_marker_lookup(
        marker_lookup=lookup,
        taxonomy_tree=taxonomy_tree_fixture,
        query_gene_names=query_gene_fixture)

    # test case where a parent is not listed in the lookup
    for to_drop in ('None', 'class/A', 'subclass/d'):
        lookup = copy.deepcopy(complete_lookup_fixture)
        lookup.pop(to_drop)
        with pytest.raises(RuntimeError, match="not listed in marker lookup"):
            validate_marker_lookup(
                marker_lookup=lookup,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=query_gene_fixture)

    # test case where a parent has no valid markers in query gene set
    for to_drop in ('None', ):
        lookup = copy.deepcopy(complete_lookup_fixture)
        lookup.pop(to_drop)
        lookup[to_drop] = [f'j_{ii}' for ii in range(3)]
        with pytest.raises(RuntimeError, match="has no valid markers in query"):
            validate_marker_lookup(
                marker_lookup=lookup,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=query_gene_fixture)

    # test patching of missing markers
    for to_drop in ('class/A', 'subclass/f'):
        lookup = copy.deepcopy(complete_lookup_fixture)
        lookup.pop(to_drop)
        lookup[to_drop] = [f'j_{ii}' for ii in range(3)]
        validate_marker_lookup(
                marker_lookup=lookup,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=query_gene_fixture)

        if to_drop == 'class/A':
            assert lookup['class/A'] == lookup['None']
        else:
            assert lookup['subclass/f'] == lookup['class/C']

    # test case where we have to skip two levels to get markers
    lookup = copy.deepcopy(complete_lookup_fixture)
    nonsense = [f'j_{ii}' for ii in range(4)]
    lookup['class/C'] = nonsense
    lookup['subclass/f'] = nonsense

    validate_marker_lookup(
                marker_lookup=lookup,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=query_gene_fixture)

    assert lookup['class/C'] == lookup['None']
    assert lookup['subclass/f'] == lookup['None']
    assert lookup['subclass/e'] != lookup['None']

    # test case where a parent has entry in lookup, but is blank
    for to_drop in ('None', 'class/A', 'subclass/d'):
        lookup = copy.deepcopy(complete_lookup_fixture)
        lookup.pop(to_drop)
        lookup[to_drop] = []
        with pytest.raises(RuntimeError,
                           match="has no valid markers in marker_lookup"):
            validate_marker_lookup(
                marker_lookup=lookup,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=query_gene_fixture)

    # test all failures together
    lookup = copy.deepcopy(complete_lookup_fixture)
    lookup.pop('class/A')
    lookup['subclass/d'] = []
    lookup['None'] = ['x', 'y', 'z']
    with pytest.raises(RuntimeError, match='validating marker lookup'):
        validate_marker_lookup(
            marker_lookup=lookup,
            taxonomy_tree=taxonomy_tree_fixture,
            query_gene_names=query_gene_fixture)
