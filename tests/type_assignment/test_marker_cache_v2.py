import pytest

import copy
import numpy as np
import warnings

from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree

from cell_type_mapper.cli.cli_log import CommandLog

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
        'cluster': {str(ii): [] for ii in range(1, 13, 1)}
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # test that valid case passes
        new_lookup = validate_marker_lookup(
            marker_lookup=complete_lookup_fixture,
            taxonomy_tree=taxonomy_tree_fixture,
            query_gene_names=query_gene_fixture)

        new_lookup == complete_lookup_fixture

        # test that it is okay to miss trivial parents
        lookup = copy.deepcopy(complete_lookup_fixture)
        lookup.pop('class/B')
        lookup['subclass/e'] = []
        lookup['subclass/b'] = ['x', 'y', 'z']
        new_lookup = validate_marker_lookup(
            marker_lookup=lookup,
            taxonomy_tree=taxonomy_tree_fixture,
            query_gene_names=query_gene_fixture)

        new_lookup == lookup

        # test case where a parent is not listed in the lookup
        for to_drop in ('None', 'class/A', 'subclass/d'):
            lookup = copy.deepcopy(complete_lookup_fixture)
            lookup.pop(to_drop)
            msg = "not listed in marker lookup"
            if to_drop == 'None':
                with pytest.raises(RuntimeError, match=msg):
                    validate_marker_lookup(
                        marker_lookup=lookup,
                        taxonomy_tree=taxonomy_tree_fixture,
                        query_gene_names=query_gene_fixture)
            else:
                with pytest.warns(UserWarning, match=msg):
                    validate_marker_lookup(
                        marker_lookup=lookup,
                        taxonomy_tree=taxonomy_tree_fixture,
                        query_gene_names=query_gene_fixture)

        # test case where a parent has no valid markers in query gene set
        for to_drop in ('None', ):
            lookup = copy.deepcopy(complete_lookup_fixture)
            lookup.pop(to_drop)
            lookup[to_drop] = [f'j_{ii}' for ii in range(3)]
            with pytest.raises(RuntimeError,
                               match="has no valid markers in query"):
                validate_marker_lookup(
                    marker_lookup=lookup,
                    taxonomy_tree=taxonomy_tree_fixture,
                    query_gene_names=query_gene_fixture)

        # test patching of missing markers
        for to_drop in ('class/A', 'subclass/f'):
            lookup = copy.deepcopy(complete_lookup_fixture)
            lookup.pop(to_drop)
            lookup[to_drop] = [f'j_{ii}' for ii in range(3)]
            log = CommandLog()
            new_lookup = validate_marker_lookup(
                marker_lookup=lookup,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=query_gene_fixture,
                log=log,
                min_markers=1)

            if to_drop == 'class/A':
                assert set(new_lookup['class/A']) == set(lookup['None'])
            else:
                assert set(new_lookup['subclass/f']) == set(lookup['class/C'])
            expected = (
                f"'{to_drop}' had too few markers "
                "in query set; augmenting"
            )
            found = False
            for msg in log.log:
                if expected in msg:
                    found = True
                    break
            assert found

        # test case where we have to skip two levels to get markers
        lookup = copy.deepcopy(complete_lookup_fixture)
        nonsense = [f'j_{ii}' for ii in range(4)]
        lookup['class/C'] = nonsense
        lookup['subclass/f'] = nonsense
        log = CommandLog()

        new_lookup = validate_marker_lookup(
                marker_lookup=lookup,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=query_gene_fixture,
                log=log)

        assert set(new_lookup['class/C']) == set(lookup['None'])
        assert set(new_lookup['subclass/f']) == set(lookup['None'])
        assert set(new_lookup['subclass/e']) != set(lookup['None'])

        for to_drop in ('class/C', 'subclass/f'):
            expected = (f"'{to_drop}' had too few markers in query set; "
                        "augmenting with markers from")
            found = False
            for msg in log.log:
                if expected in msg:
                    found = True
                    break
            assert found

        # test case where a parent has entry in lookup, but is blank
        for to_drop in ('None', 'class/A', 'subclass/d'):
            lookup = copy.deepcopy(complete_lookup_fixture)
            lookup.pop(to_drop)
            lookup[to_drop] = []
            msg = "has no valid markers in marker_lookup"
            if to_drop == 'None':
                with pytest.raises(RuntimeError, match=msg):
                    validate_marker_lookup(
                        marker_lookup=lookup,
                        taxonomy_tree=taxonomy_tree_fixture,
                        query_gene_names=query_gene_fixture)
            else:
                with pytest.warns(UserWarning, match=msg):
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

        # test case where there are no markers at all
        msg = (
            "After comparing query data to reference data, no valid marker "
            "genes could be found at any level"
        )
        with pytest.raises(RuntimeError, match=msg):
            validate_marker_lookup(
                marker_lookup=complete_lookup_fixture,
                taxonomy_tree=taxonomy_tree_fixture,
                query_gene_names=['bad', 'gene', 'name', 'list'])


def test_patching_of_marker_lookup():
    """
    Test simple cases of patching up taxonomic nodes that have
    too few markers.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree_data = {
            'hierarchy': ['class', 'subclass', 'cluster'],
            'class': {
                'A': ['a', 'b'],
                'B': ['c', 'd', 'e']
            },
            'subclass': {
                'a': ['1', '2', '3'],
                'b': ['4'],
                'c': ['5', '6'],
                'd': ['7', '8'],
                'e': ['9', '10']
            },
            'cluster': {
                str(k): [] for k in range(1, 11, 1)
            }
        }

        taxonomy_tree = TaxonomyTree(data=taxonomy_tree_data)

        marker_lookup = {
            'None': ['g0', 'g1', 'g2'],
            'class/A': ['g3', 'g4'],
            'class/B': ['g5'],
            'subclass/a': ['g6', 'g7'],
            'subclass/d': [],
            'subclass/e': ['g8']
        }

        query_gene_names = [f'g{ii}' for ii in range(9)]
        actual = validate_marker_lookup(
            marker_lookup=marker_lookup,
            query_gene_names=query_gene_names,
            taxonomy_tree=taxonomy_tree,
            min_markers=1)

        expected = {
            'None': ['g0', 'g1', 'g2'],
            'class/A': ['g3', 'g4'],
            'class/B': ['g5'],
            'subclass/a': ['g6', 'g7'],
            'subclass/c': ['g5'],
            'subclass/d': ['g5'],
            'subclass/e': ['g8']
        }

        assert actual == expected
        assert marker_lookup != expected

        actual = validate_marker_lookup(
            marker_lookup=marker_lookup,
            query_gene_names=query_gene_names,
            taxonomy_tree=taxonomy_tree,
            min_markers=2)

        expected = {
            'None': ['g0', 'g1', 'g2'],
            'class/A': ['g3', 'g4'],
            'class/B': ['g0', 'g1', 'g2', 'g5'],
            'subclass/a': ['g6', 'g7'],
            'subclass/c': ['g0', 'g1', 'g2', 'g5'],
            'subclass/d': ['g0', 'g1', 'g2', 'g5'],
            'subclass/e': ['g5', 'g8']
        }

        assert actual == expected
        assert marker_lookup != expected
