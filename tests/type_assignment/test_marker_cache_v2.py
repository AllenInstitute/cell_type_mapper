import pytest

import copy
import h5py
import json
import numpy as np
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree

from cell_type_mapper.cli.cli_log import CommandLog

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    validate_marker_lookup,
    create_marker_cache_from_specified_markers)

from cell_type_mapper.type_assignment.matching import (
    assemble_query_data,
    assemble_query_data_hann
)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix
)


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


@pytest.fixture
def marker_cache_fixture(tmp_dir_fixture):
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
        taxonomy_tree = TaxonomyTree(data=tree_data)

    reference_gene_names = [
        'g0', 'g9', 'g3', 'g1', 'g5', 'g4', 'g8',
        'g6', 'g7', 'g2'
    ]
    query_gene_names = [f'g{ii}' for ii in range(5, 20, 1)]
    marker_lookup = {
        'None': ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7'],
        'class/A': ['g0', 'g5', 'g6', 'g8'],
        'subclass/b': ['g1', 'g7', 'g8', 'g9'],
        'subclass/c': ['g3', 'g4', 'g6', 'g7', 'g9']
    }
    cache_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_cache_',
        suffix='.h5'
    )
    create_marker_cache_from_specified_markers(
        marker_lookup=marker_lookup,
        reference_gene_names=reference_gene_names,
        query_gene_names=query_gene_names,
        output_cache_path=cache_path,
        taxonomy_tree=taxonomy_tree,
        min_markers=2
    )
    return {
        'cache_path': cache_path,
        'reference_gene_names': reference_gene_names,
        'query_gene_names': query_gene_names,
        'taxonomy_tree': taxonomy_tree
    }


def test_create_marker_cache_from_specified_markers(
        marker_cache_fixture):
    """
    Make sure that marker gene data is transcribed correctly
    to hdf5 cache.
    """

    cache_path = marker_cache_fixture['cache_path']
    reference_gene_names = marker_cache_fixture['reference_gene_names']
    query_gene_names = marker_cache_fixture['query_gene_names']

    with h5py.File(cache_path, 'r') as src:
        assert set(src.keys()) == set(['all_reference_markers',
                                       'all_query_markers',
                                       'query_gene_names',
                                       'reference_gene_names',
                                       'None',
                                       'subclass',
                                       'class',
                                       'parent_node_list'])

        assert list(src['class'].keys()) == ['A']
        assert set(src['subclass'].keys()) == set(['b', 'c'])

        parent_node_list = json.loads(
            src['parent_node_list'][()].decode('utf-8')
        )

        all_ref = src['all_reference_markers'][()]
        all_query = src['all_query_markers'][()]
        actual_query_gene_names = json.loads(
            src['query_gene_names'][()].decode('utf-8')
        )
        actual_reference_gene_names = json.loads(
            src['reference_gene_names'][()].decode('utf-8')
        )

        np.testing.assert_array_equal(
            src['None']['reference'][()],
            np.array([4, 7, 8])
        )
        np.testing.assert_array_equal(
            src['None']['query'][()],
            np.array([0, 1, 2])
        )

        np.testing.assert_array_equal(
            src['class/A']['reference'][()],
            np.array([4, 6, 7])
        )
        np.testing.assert_array_equal(
            src['class/A']['query'][()],
            np.array([0, 3, 1])
        )

        np.testing.assert_array_equal(
            src['subclass/b']['reference'][()],
            np.array([1, 6, 8])
        )
        np.testing.assert_array_equal(
            src['subclass/b']['query'][()],
            np.array([4, 3, 2])
        )

        np.testing.assert_array_equal(
            src['subclass/c']['reference'][()],
            np.array([1, 7, 8])
        )
        np.testing.assert_array_equal(
            src['subclass/c']['query'][()],
            np.array([4, 1, 2])
        )

    np.testing.assert_array_equal(
        actual_query_gene_names,
        np.array(query_gene_names)
    )

    np.testing.assert_array_equal(
        actual_reference_gene_names,
        np.array(reference_gene_names)
    )

    np.testing.assert_array_equal(
        all_ref, np.array([1, 4, 6, 7, 8])
    )
    np.testing.assert_array_equal(
        all_query, np.array([0, 1, 2, 3, 4])
    )

    np.testing.assert_array_equal(
        np.array(parent_node_list),
        np.array([
            'None',
            'class/A',
            'subclass/b',
            'subclass/c'
        ])
    )


def test_assemble_query_data_from_file(
        marker_cache_fixture):

    rng = np.random.default_rng(13111)
    n_query_cells = 20
    query_cells = [f'c{ii}' for ii in range(n_query_cells)]
    query_genes = marker_cache_fixture['query_gene_names']
    raw_query_data = rng.random((len(query_cells), len(query_genes)))
    query_data = CellByGeneMatrix(
        data=np.copy(raw_query_data),
        gene_identifiers=query_genes,
        normalization='log2CPM',
        cell_identifiers=query_cells
    )

    taxonomy_tree = marker_cache_fixture['taxonomy_tree']
    reference_genes = marker_cache_fixture['reference_gene_names']
    cluster_list = taxonomy_tree.nodes_at_level(taxonomy_tree.leaf_level)
    raw_reference_data = rng.random((len(cluster_list), len(reference_genes)))
    reference_data = CellByGeneMatrix(
        data=np.copy(raw_reference_data),
        gene_identifiers=reference_genes,
        normalization='log2CPM',
        cell_identifiers=cluster_list
    )

    result = assemble_query_data(
        full_query_data=query_data,
        mean_profile_matrix=reference_data,
        taxonomy_tree=taxonomy_tree,
        marker_cache_path=marker_cache_fixture['cache_path'],
        parent_node=('class', 'A')
    )

    for key in ('reference_data', 'query_data'):
        np.testing.assert_array_equal(
            np.array(result[key].gene_identifiers),
            np.array(['g5', 'g8', 'g6'])
        )

    np.testing.assert_array_equal(
        np.array(result['reference_data'].cell_identifiers),
        np.array(['a1', 'b1', 'b2'])
    )

    expected = raw_reference_data[
        np.array([0, 1, 2]), :][:, np.array([4, 6, 7])]

    np.testing.assert_allclose(
        expected,
        result['reference_data'].data,
        atol=0.0,
        rtol=1.0e-6
    )

    np.testing.assert_array_equal(
        np.array(result['query_data'].cell_identifiers),
        np.array([f'c{ii}' for ii in range(n_query_cells)])
    )

    expected = raw_query_data[:, np.array([0, 3, 1])]
    np.testing.assert_allclose(
        expected,
        result['query_data'].data,
        atol=0.0,
        rtol=1.0e-6
    )

    np.testing.assert_array_equal(
        np.array(result['reference_types']),
        np.array(['a', 'b', 'b'])
    )


def test_assemble_query_data_hann(
        marker_cache_fixture):
    """
    Test the assemble_query_data function for the HANN algorithm
    """

    rng = np.random.default_rng(7788112)
    n_query_cells = 20
    query_cells = [f'c{ii}' for ii in range(n_query_cells)]
    query_genes = marker_cache_fixture['query_gene_names']
    raw_query_data = rng.random((len(query_cells), len(query_genes)))
    query_data = CellByGeneMatrix(
        data=np.copy(raw_query_data),
        gene_identifiers=query_genes,
        normalization='log2CPM',
        cell_identifiers=query_cells
    )

    taxonomy_tree = marker_cache_fixture['taxonomy_tree']
    reference_genes = marker_cache_fixture['reference_gene_names']
    cluster_list = taxonomy_tree.nodes_at_level(taxonomy_tree.leaf_level)
    raw_reference_data = rng.random((len(cluster_list), len(reference_genes)))
    reference_data = CellByGeneMatrix(
        data=np.copy(raw_reference_data),
        gene_identifiers=reference_genes,
        normalization='log2CPM',
        cell_identifiers=cluster_list
    )

    result = assemble_query_data_hann(
        full_query_data=query_data,
        mean_profile_matrix=reference_data,
        taxonomy_tree=taxonomy_tree,
        marker_cache_path=marker_cache_fixture['cache_path']
    )

    for key in ('reference_data', 'query_data'):
        np.testing.assert_array_equal(
            np.array(result[key].gene_identifiers),
            np.array(['g5', 'g6', 'g7', 'g8', 'g9'])
        )

    np.testing.assert_array_equal(
        np.array(result['reference_data'].cell_identifiers),
        np.array(['a1', 'b1', 'b2', 'c1', 'c2'])
    )

    expected = raw_reference_data[:, np.array([4, 7, 8, 6, 1])]
    np.testing.assert_allclose(
        result['reference_data'].data,
        expected,
        atol=0.0,
        rtol=1.0-6
    )

    np.testing.assert_array_equal(
        np.array(result['query_data'].cell_identifiers),
        np.array([f'c{ii}' for ii in range(n_query_cells)])
    )

    expected = raw_query_data[:, np.arange(5, dtype=int)]
    np.testing.assert_allclose(
        expected,
        result['query_data'].data,
        atol=0.0,
        rtol=1.0e-6
    )

    expected_markers = {
        'None': [0, 1, 2],
        'class/A': [0, 1, 3],
        'subclass/b': [2, 3, 4],
        'subclass/c': [1, 2, 4]
    }
    actual_markers = result['marker_lookup']
    assert set(expected_markers.keys()) == set(actual_markers.keys())
    for key in expected_markers.keys():
        np.testing.assert_array_equal(
            np.array(actual_markers[key]),
            np.array(expected_markers[key])
        )
