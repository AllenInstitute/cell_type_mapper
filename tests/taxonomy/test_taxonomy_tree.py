import pytest
import copy
import json
import itertools
import re
import warnings

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def test_tree_as_leaves(
        records_fixture,
        column_hierarchy,
        parent_to_leaves,
        taxonomy_tree_fixture):

    as_leaves = taxonomy_tree_fixture.as_leaves

    for h in column_hierarchy:
        for this_node in parent_to_leaves[h]:
            expected = parent_to_leaves[h][this_node]
            actual = as_leaves[h][this_node]
            assert set(actual) == expected
            assert len(set(actual)) == len(actual)


def test_tree_get_all_pairs(
        records_fixture,
        column_hierarchy,
        l1_to_l2_fixture,
        l2_to_class_fixture,
        class_to_cluster_fixture,
        taxonomy_tree_fixture):

    siblings = taxonomy_tree_fixture.siblings
    ct = 0
    for level, lookup in zip(('level1', 'level2', 'class'),
                             (l1_to_l2_fixture[0],
                              l2_to_class_fixture[0],
                              class_to_cluster_fixture[0])):
        elements = list(lookup.keys())
        elements.sort()
        for i0 in range(len(elements)):
            for i1 in range(i0+1, len(elements), 1):
                test = (level, elements[i0], elements[i1])
                assert test in siblings
                ct += 1
    cluster_list = []
    for k in class_to_cluster_fixture[0]:
        cluster_list += list(class_to_cluster_fixture[0][k])
    cluster_list.sort()
    for i0 in range(len(cluster_list)):
        for i1 in range(i0+1, len(cluster_list), 1):
            test = ('cluster', cluster_list[i0], cluster_list[i1])
            assert test in siblings
            ct += 1
    assert len(siblings) == ct


def test_tree_get_all_leaf_pairs():

    tree_data = {
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': set(['l2b', 'l2d']),
                   'l1b': set(['l2a', 'l2c', 'l2e']),
                   'l1c': set(['l2f',])
                   },
        'level2': {'l2a': set(['l3b',]),
                   'l2b': set(['l3a', 'l3c']),
                   'l2c': set(['l3e',]),
                   'l2d': set(['l3d', 'l3f', 'l3h']),
                   'l2e': set(['l3g',]),
                   'l2f': set(['l3i',])},
        'level3': {'l3a': set([str(ii) for ii in range(3)]),
                   'l3b': set([str(ii) for ii in range(3, 7)]),
                   'l3c': set([str(ii) for ii in range(7, 9)]),
                   'l3d': set([str(ii) for ii in range(9, 13)]),
                   'l3e': set([str(ii) for ii in range(13, 15)]),
                   'l3f': set([str(ii) for ii in range(15, 19)]),
                   'l3g': set([str(ii) for ii in range(19, 21)]),
                   'l3h': set([str(ii) for ii in range(21, 23)]),
                   'l3i': set(['23',])},
        'leaf': {str(k): range(26*k, 26*(k+1))
                 for k in range(24)}}

    taxonomy_tree = TaxonomyTree(data=tree_data)

    # check non-None parent Node
    parent_node = ('level2', 'l2b')
    actual = taxonomy_tree.leaves_to_compare(parent_node)

    candidates0 = ['0', '1', '2']
    candidates1 = ['7', '8']
    expected = []
    for pair in itertools.product(candidates0, candidates1):
        expected.append(('leaf', pair[0], pair[1]))

    assert set(actual) == set(expected)

    # check case with parent_node = None
    parent_node = None
    candidates0 = ['0', '1', '2', '7', '8', '9', '10', '11', '12',
                   '15', '16', '17', '18', '21', '22']
    candidates1 = ['3', '4', '5', '6', '13', '14', '19', '20']
    candidates2 = ['23']

    actual = taxonomy_tree.leaves_to_compare(parent_node)

    expected = []
    for c0, c1 in itertools.combinations([candidates0,
                                          candidates1,
                                          candidates2], 2):
        for pair in itertools.product(c0, c1):
            if pair[0] < pair[1]:
                expected.append(('leaf', pair[0], pair[1]))
            else:
                expected.append(('leaf', pair[1], pair[0]))
    assert set(actual) == set(expected)

    # check case when there are no pairs to compare
    parent_node = ('level1', 'l1c')
    actual = taxonomy_tree.leaves_to_compare(parent_node)
    assert actual == []

    parent_node = ('leaf', '15')
    actual = taxonomy_tree.leaves_to_compare(parent_node)
    assert actual == []


def test_tree_eq():
    """
    Test implementations of __eq__ and __ne__ in TaxonomyTree
    """
    data1 = {
        'hierarchy': ['a', 'b'],
        'a': {'aa': ['aaa', 'bbb'],
              'bb': ['ccc', 'ddd']},
        'b': {'aaa': [1, 2],
              'bbb': [3],
              'ccc': [4, 5],
              'ddd': [6]}}

    tree1 = TaxonomyTree(data=data1)
    tree2 = TaxonomyTree(data=copy.deepcopy(data1))
    assert tree1 == tree2
    assert not tree1 != tree2
    assert tree1 is not tree2

    data2 = {
        'hierarchy': ['a', 'd'],
        'a': {'aa': ['aaa', 'bbb'],
              'bb': ['ccc', 'ddd']},
        'd': {'aaa': [1, 2],
              'bbb': [3],
              'ccc': [4, 5],
              'ddd': [6]}}

    tree3 = TaxonomyTree(data=data2)
    assert not tree3 == tree1
    assert tree3 != tree1


def test_flattening_tree():
    tree_data = {
        'metadata': {'hello': 'there'},
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': set(['l2b', 'l2d']),
                   'l1b': set(['l2a', 'l2c', 'l2e']),
                   'l1c': set(['l2f',])
                   },
        'level2': {'l2a': set(['l3b',]),
                   'l2b': set(['l3a', 'l3c']),
                   'l2c': set(['l3e',]),
                   'l2d': set(['l3d', 'l3f', 'l3h']),
                   'l2e': set(['l3g',]),
                   'l2f': set(['l3i',])},
        'level3': {'l3a': set([str(ii) for ii in range(3)]),
                   'l3b': set([str(ii) for ii in range(3, 7)]),
                   'l3c': set([str(ii) for ii in range(7, 9)]),
                   'l3d': set([str(ii) for ii in range(9, 13)]),
                   'l3e': set([str(ii) for ii in range(13, 15)]),
                   'l3f': set([str(ii) for ii in range(15, 19)]),
                   'l3g': set([str(ii) for ii in range(19, 21)]),
                   'l3h': set([str(ii) for ii in range(21, 23)]),
                   'l3i': set(['23',])},
        'leaf': {str(k): list(range(26*k, 26*(k+1)))
                 for k in range(24)}}

    first_tree = TaxonomyTree(data=tree_data)
    flat_tree = first_tree.flatten()
    assert isinstance(flat_tree, TaxonomyTree)
    assert flat_tree.hierarchy == ['leaf']
    assert flat_tree._data['metadata'] == {'hello': 'there', 'flattened': True}
    assert flat_tree.leaf_level == 'leaf'
    assert flat_tree.all_leaves == first_tree.all_leaves
    as_leaves = flat_tree.as_leaves
    assert len(as_leaves) == 1
    assert list(as_leaves.keys()) == ['leaf']
    assert flat_tree.all_parents == [None]

    flat_leaves = flat_tree.nodes_at_level('leaf')
    base_leaves = first_tree.nodes_at_level('leaf')
    assert flat_leaves == base_leaves
    assert first_tree.hierarchy == ['level1', 'level2', 'level3', 'leaf']
    for node in flat_leaves:
        assert (
            first_tree.children('leaf', node)
            == flat_tree.children('leaf', node)
        )


def test_drop_level():
    """
    Test fuction to drop a level from the tree
    """
    data = {
        'hierarchy': ['l0', 'l1', 'l2', 'l3'],
        'l0': {
            'a0': ['b0', 'b3'],
            'a1': ['b1', 'b4'],
            'a2': ['b2']
        },
        'l1': {
            'b0': ['c1', 'c5'],
            'b1': ['c2'],
            'b2': ['c0', 'c4', 'c3'],
            'b3': ['c6'],
            'b4': ['c7', 'c8']
        },
        'l2': {
            f'c{ii}': [f'd{ii}', f'd{ii+9}']
            for ii in range(9)
        },
        'l3': {
            f'd{ii}': list(range(ii*10, 5+ii*10))
            for ii in range(18)
        }
    }

    baseline_tree = TaxonomyTree(data=data)

    with pytest.raises(RuntimeError, match='That is the leaf level'):
        baseline_tree.drop_level('l3')

    with pytest.raises(RuntimeError, match='not in the hierarchy'):
        baseline_tree.drop_level('xyz')

    flat_tree = baseline_tree.flatten()
    with pytest.raises(RuntimeError, match='It is flat'):
        flat_tree.drop_level('l1')

    dropped_l0 = {
        'hierarchy': ['l1', 'l2', 'l3'],
        'l1': {
            'b0': ['c1', 'c5'],
            'b1': ['c2'],
            'b2': ['c0', 'c4', 'c3'],
            'b3': ['c6'],
            'b4': ['c7', 'c8']
        },
        'l2': {
            f'c{ii}': [f'd{ii}', f'd{ii+9}']
            for ii in range(9)
        },
        'l3': {
            f'd{ii}': list(range(ii*10, 5+ii*10))
            for ii in range(18)
        }
    }

    expected = TaxonomyTree(data=dropped_l0)
    actual = baseline_tree.drop_level('l0')
    assert expected == actual
    assert not baseline_tree == actual

    dropped_l1 = {
        'hierarchy': ['l0', 'l2', 'l3'],
        'l0': {
            'a0': ['c1', 'c5', 'c6'],
            'a1': ['c2', 'c7', 'c8'],
            'a2': ['c0', 'c4', 'c3']
        },
        'l2': {
            f'c{ii}': [f'd{ii}', f'd{ii+9}']
            for ii in range(9)
        },
        'l3': {
            f'd{ii}': list(range(ii*10, 5+ii*10))
            for ii in range(18)
        }
    }

    expected = TaxonomyTree(data=dropped_l1)
    actual = baseline_tree.drop_level('l1')
    assert actual == expected
    assert not baseline_tree == actual

    dropped_l2 = {
        'hierarchy': ['l0', 'l1', 'l3'],
        'l0': {
            'a0': ['b0', 'b3'],
            'a1': ['b1', 'b4'],
            'a2': ['b2']
        },
        'l1': {
            'b0': ['d1', 'd10', 'd5', 'd14'],
            'b1': ['d2', 'd11'],
            'b2': ['d0', 'd9', 'd4', 'd13', 'd3', 'd12'],
            'b3': ['d6', 'd15'],
            'b4': ['d7', 'd16', 'd8', 'd17']
        },
        'l3': {
            f'd{ii}': list(range(ii*10, 5+ii*10))
            for ii in range(18)
        }
    }
    expected = TaxonomyTree(data=dropped_l2)
    actual = baseline_tree.drop_level('l2')
    assert actual == expected
    assert not baseline_tree == actual

    # test dropping leaf level
    dropped_l3 = {
        'hierarchy': ['l0', 'l1', 'l2'],
        'l0': {
            'a0': ['b0', 'b3'],
            'a1': ['b1', 'b4'],
            'a2': ['b2']
        },
        'l1': {
            'b0': ['c1', 'c5'],
            'b1': ['c2'],
            'b2': ['c0', 'c4', 'c3'],
            'b3': ['c6'],
            'b4': ['c7', 'c8']
        },
        'l2': {
            f'c{ii}': (list(range(ii*10, 5+ii*10))
                       + list(range(10*(ii+9), 5+(ii+9)*10)))
            for ii in range(9)
        }
    }
    expected = TaxonomyTree(data=dropped_l3)
    actual = baseline_tree.drop_leaf_level()
    assert actual.leaf_level == 'l2'
    assert actual == expected
    assert not baseline_tree == actual


def test_parents_method():
    """
    Test method that returns the parents (all of them) of a node
    """
    data = {
        'hierarchy': ['a', 'b', 'c'],
        'a': {
            'aa': ['1', '2'],
            'bb': ['3']
        },
        'b': {
            '1': ['x'],
            '2': ['y', 'z'],
            '3': ['w', 'u', 'v']
        },
        'c': {
            'u': [],
            'v': [],
            'w': [],
            'x': [],
            'y': [],
            'z': []
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(data=data)

    actual = taxonomy_tree.parents('c', 'u')
    expected = {'b': '3', 'a': 'bb'}
    assert actual == expected
    actual = taxonomy_tree.parents('b', '2')
    expected = {'a': 'aa'}
    assert actual == expected


@pytest.mark.parametrize("drop_cells", (True, False))
def test_tree_to_str(drop_cells):
    tree_data = {
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': list(['l2b', 'l2d']),
                   'l1b': list(['l2a', 'l2c', 'l2e']),
                   'l1c': list(['l2f',])
                   },
        'level2': {'l2a': ['l3b',],
                   'l2b': ['l3a', 'l3c'],
                   'l2c': ['l3e',],
                   'l2d': ['l3d', 'l3f', 'l3h'],
                   'l2e': ['l3g',],
                   'l2f': ['l3i',]},
        'level3': {'l3a': [str(ii) for ii in range(3)],
                   'l3b': [str(ii) for ii in range(3, 7)],
                   'l3c': [str(ii) for ii in range(7, 9)],
                   'l3d': [str(ii) for ii in range(9, 13)],
                   'l3e': [str(ii) for ii in range(13, 15)],
                   'l3f': [str(ii) for ii in range(15, 19)],
                   'l3g': [str(ii) for ii in range(19, 21)],
                   'l3h': [str(ii) for ii in range(21, 23)],
                   'l3i': ['23',]},
        'leaf': {str(k): list(range(26*k, 26*(k+1)))
                 for k in range(24)}}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(data=tree_data)

    tree_str = taxonomy_tree.to_str(drop_cells=drop_cells)

    tree_str_rehydrated = json.loads(tree_str)
    assert set(tree_data.keys()) == set(tree_str_rehydrated.keys())
    for k in tree_data:
        if drop_cells and k == taxonomy_tree.leaf_level:
            assert (
                set(tree_data[k].keys())
                == set(tree_str_rehydrated[k].keys())
            )
            for leaf in tree_str_rehydrated[k]:
                assert tree_str_rehydrated[k][leaf] == []
        else:
            assert tree_data[k] == tree_str_rehydrated[k]

    # make sure tree was not changed in place
    for leaf in tree_data['leaf']:
        expected = tree_data['leaf'][leaf]
        assert taxonomy_tree.children(level='leaf', node=leaf) == expected

    # try re-instantiating tree without cells

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        new_tree = TaxonomyTree(data=json.loads(tree_str))

    assert set(new_tree.all_leaves) == set(taxonomy_tree.all_leaves)


def test_backfill():
    """
    Test the method to backfill assignments
    """

    data = {
        "hierarchy": ["A", "B", "C", "D"],
        "A": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in ('a', 'b', 'c')
        },
        "B": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in ('aa', 'ab', 'ba', 'bb', 'ca', 'cb')
        },
        "C": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in (
                'aaa', 'aab', 'aba', 'abb', 'baa', 'bab',
                'bba', 'bbb', 'caa', 'cab', 'cba', 'cbb')
        },
        "D": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in (
               'aaaa', 'aaab', 'aaba', 'aabb',
               'abaa', 'abab', 'abba', 'abbb',
               'baaa', 'baab', 'baba', 'babb',
               'bbaa', 'bbab', 'bbba', 'bbbb',
               'caaa', 'caab', 'caba', 'cabb',
               'cbaa', 'cbab', 'cbba', 'cbbb')
        }
    }

    tree = TaxonomyTree(data=data)

    expected_assignment = [
        {'A': {'assignment': 'b', 'data': 2},
         'B': {'assignment': 'bb', 'other': 'sure'},
         'C': {'assignment': 'bba', 'other': 'sure'},
         'D': {'assignment': 'bbaa', 'this': 'is'},
         'id': 'alice'
         },
        {'A': {'assignment': 'c', 'junk': 'garbage'},
         'B': {'assignment': 'ca', 'junk': 'garbage'},
         'C': {'assignment': 'cab', 'silly': 'walk'},
         'D': {'assignment': 'caba'},
         'something': 'else'
         },
        {'A': {'assignment': 'a'},
         'B': {'assignment': 'ab'},
         'C': {'assignment': 'aba', 'to': 'be'},
         'D': {'assignment': 'abab', 'to': 'be'},
         'why': 'not'
         },
        {'A': {'assignment': 'a'},
         # backfill should not fix if child missing
         'B': {'assignment': 'aa'},
         'because': 'so'
         },
        {'A': {'assignment': 'b'},
         'B': {'assignment': 'bb'},
         'C': {'assignment': 'bba'},
         # backfill should not fix if child missing
         'that': 'is true'
         },
        {'A': {'assignment': 'b'},
         'B': {'assignment': 'ba'},
         'C': {'assignment': 'baa'},
         'D': {'assignment': 'baaa'},
         'that': 'is false'
         },
        {'A': {'assignment': 'c', 'say': 'so'},  # pop many parents
         'B': {'assignment': 'cb', 'say': 'so'},
         'C': {'assignment': 'cba',
               'say': 'so',
               'runner_up_garbage': 'uh huh'},
         'D': {'assignment': 'cbaa'},
         'that': 'is false'
         }
    ]

    noisy = copy.deepcopy(expected_assignment)

    # pop some data from noisy; edit expected to reflect that those
    # cell, level combinations will be backfilled
    for cell_idx, level in ((0, 'B'),
                            (1, 'A'),
                            (2, 'C'),
                            (6, 'A'),
                            (6, 'B')):
        noisy[cell_idx].pop(level)
        expected_assignment[cell_idx][level]['directly_assigned'] = False

    assert noisy != expected_assignment
    noisy = tree.backfill_assignments(noisy)
    assert noisy == expected_assignment


def test_bad_level():
    """
    Test errors raised when you ask for a level that does not exist
    """

    data = {
        "hierarchy": ["A", "B", "C", "D"],
        "A": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in ('a', 'b', 'c')
        },
        "B": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in ('aa', 'ab', 'ba', 'bb', 'ca', 'cb')
        },
        "C": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in (
                'aaa', 'aab', 'aba', 'abb', 'baa', 'bab',
                'bba', 'bbb', 'caa', 'cab', 'cba', 'cbb')
        },
        "D": {
            k: [f'{k}{j}' for j in ('a', 'b')]
            for k in (
               'aaaa', 'aaab', 'aaba', 'aabb',
               'abaa', 'abab', 'abba', 'abbb',
               'baaa', 'baab', 'baba', 'babb',
               'bbaa', 'bbab', 'bbba', 'bbbb',
               'caaa', 'caab', 'caba', 'cabb',
               'cbaa', 'cbab', 'cbba', 'cbbb')
        }
    }

    tree = TaxonomyTree(data=data)

    with pytest.raises(RuntimeError, match="F is not a valid level"):
        tree.nodes_at_level('F')

    with pytest.raises(RuntimeError, match="F is not a valid level"):
        tree.children(level='F', node='ffff')


@pytest.mark.parametrize(
    "node_to_drop",
    [('level1', 'l1b'),
     ('level2', 'l2d'),
     ('level3', 'l3f'),
     ('level2', 'l2f')
     ]
)
def test_drop_node(node_to_drop):

    full_tree_data = {
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': set(['l2b', 'l2d']),
                   'l1b': set(['l2a', 'l2c', 'l2e']),
                   'l1c': set(['l2f',])
                   },
        'level2': {'l2a': set(['l3b',]),
                   'l2b': set(['l3a', 'l3c']),
                   'l2c': set(['l3e',]),
                   'l2d': set(['l3d', 'l3f', 'l3h']),
                   'l2e': set(['l3g',]),
                   'l2f': set(['l3i',])},
        'level3': {'l3a': set([str(ii) for ii in range(3)]),
                   'l3b': set([str(ii) for ii in range(3, 7)]),
                   'l3c': set([str(ii) for ii in range(7, 9)]),
                   'l3d': set([str(ii) for ii in range(9, 13)]),
                   'l3e': set([str(ii) for ii in range(13, 15)]),
                   'l3f': set([str(ii) for ii in range(15, 19)]),
                   'l3g': set([str(ii) for ii in range(19, 21)]),
                   'l3h': set([str(ii) for ii in range(21, 23)]),
                   'l3i': set(['23',])},
        'leaf': {str(k): range(26*k, 26*(k+1))
                 for k in range(24)}}

    test_tree = TaxonomyTree(
        data=full_tree_data)

    if node_to_drop == ('level1', 'l1b'):
        leaf_list = [
            0, 1, 2,  7, 8, 9, 10, 11, 12,
            15, 16, 17, 18, 21, 22, 23
        ]
        expected_tree_data = {
            'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
            'level1': {'l1a': set(['l2b', 'l2d']),
                       'l1c': set(['l2f',])
                       },
            'level2': {'l2b': set(['l3a', 'l3c']),
                       'l2d': set(['l3d', 'l3f', 'l3h']),
                       'l2f': set(['l3i',])},
            'level3': {'l3a': set([str(ii) for ii in range(3)]),
                       'l3c': set([str(ii) for ii in range(7, 9)]),
                       'l3d': set([str(ii) for ii in range(9, 13)]),
                       'l3f': set([str(ii) for ii in range(15, 19)]),
                       'l3h': set([str(ii) for ii in range(21, 23)]),
                       'l3i': set(['23',])},
            'leaf': {str(k): range(26*k, 26*(k+1))
                     for k in leaf_list}}
    elif node_to_drop == ('level2', 'l2d'):
        leaf_list = [
            0, 1, 2, 3, 4, 5, 6, 7, 8,
            13, 14, 19, 20, 23
        ]

        expected_tree_data = {
            'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
            'level1': {'l1a': set(['l2b']),
                       'l1b': set(['l2a', 'l2c', 'l2e']),
                       'l1c': set(['l2f',])
                       },
            'level2': {'l2a': set(['l3b',]),
                       'l2b': set(['l3a', 'l3c']),
                       'l2c': set(['l3e',]),
                       'l2e': set(['l3g',]),
                       'l2f': set(['l3i',])},
            'level3': {'l3a': set([str(ii) for ii in range(3)]),
                       'l3b': set([str(ii) for ii in range(3, 7)]),
                       'l3c': set([str(ii) for ii in range(7, 9)]),
                       'l3e': set([str(ii) for ii in range(13, 15)]),
                       'l3g': set([str(ii) for ii in range(19, 21)]),
                       'l3i': set(['23',])},
            'leaf': {str(k): range(26*k, 26*(k+1))
                     for k in leaf_list}}
    elif node_to_drop == ('level3', 'l3f'):
        leaf_list = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 19, 20, 21, 22, 23
        ]

        expected_tree_data = {
            'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
            'level1': {'l1a': set(['l2b', 'l2d']),
                       'l1b': set(['l2a', 'l2c', 'l2e']),
                       'l1c': set(['l2f',])
                       },
            'level2': {'l2a': set(['l3b',]),
                       'l2b': set(['l3a', 'l3c']),
                       'l2c': set(['l3e',]),
                       'l2d': set(['l3d', 'l3h']),
                       'l2e': set(['l3g',]),
                       'l2f': set(['l3i',])},
            'level3': {'l3a': set([str(ii) for ii in range(3)]),
                       'l3b': set([str(ii) for ii in range(3, 7)]),
                       'l3c': set([str(ii) for ii in range(7, 9)]),
                       'l3d': set([str(ii) for ii in range(9, 13)]),
                       'l3e': set([str(ii) for ii in range(13, 15)]),
                       'l3g': set([str(ii) for ii in range(19, 21)]),
                       'l3h': set([str(ii) for ii in range(21, 23)]),
                       'l3i': set(['23',])},
            'leaf': {str(k): range(26*k, 26*(k+1))
                     for k in leaf_list}}
    elif node_to_drop == ('level2', 'l2f'):
        leaf_list = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22
        ]

        expected_tree_data = {
            'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
            'level1': {'l1a': set(['l2b', 'l2d']),
                       'l1b': set(['l2a', 'l2c', 'l2e']),
                       },
            'level2': {'l2a': set(['l3b',]),
                       'l2b': set(['l3a', 'l3c']),
                       'l2c': set(['l3e',]),
                       'l2d': set(['l3d', 'l3f', 'l3h']),
                       'l2e': set(['l3g',])},
            'level3': {'l3a': set([str(ii) for ii in range(3)]),
                       'l3b': set([str(ii) for ii in range(3, 7)]),
                       'l3c': set([str(ii) for ii in range(7, 9)]),
                       'l3d': set([str(ii) for ii in range(9, 13)]),
                       'l3e': set([str(ii) for ii in range(13, 15)]),
                       'l3f': set([str(ii) for ii in range(15, 19)]),
                       'l3g': set([str(ii) for ii in range(19, 21)]),
                       'l3h': set([str(ii) for ii in range(21, 23)])},
            'leaf': {str(k): range(26*k, 26*(k+1))
                     for k in leaf_list}}
    else:
        raise RuntimeError(
            f"No test for dropped_node {node_to_drop}"
        )

    expected_tree = TaxonomyTree(data=expected_tree_data)

    assert test_tree != expected_tree

    test_tree = test_tree.drop_node(
        level=node_to_drop[0],
        node=node_to_drop[1])

    assert test_tree == expected_tree


def test_drop_leaf_node():

    full_tree_data = {
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': set(['l2b', 'l2d']),
                   'l1b': set(['l2a', 'l2c', 'l2e']),
                   'l1c': set(['l2f',])
                   },
        'level2': {'l2a': set(['l3b',]),
                   'l2b': set(['l3a', 'l3c']),
                   'l2c': set(['l3e',]),
                   'l2d': set(['l3d', 'l3f', 'l3h']),
                   'l2e': set(['l3g',]),
                   'l2f': set(['l3i',])},
        'level3': {'l3a': set([str(ii) for ii in range(3)]),
                   'l3b': set([str(ii) for ii in range(3, 7)]),
                   'l3c': set([str(ii) for ii in range(7, 9)]),
                   'l3d': set([str(ii) for ii in range(9, 13)]),
                   'l3e': set([str(ii) for ii in range(13, 15)]),
                   'l3f': set([str(ii) for ii in range(15, 19)]),
                   'l3g': set([str(ii) for ii in range(19, 21)]),
                   'l3h': set([str(ii) for ii in range(21, 23)]),
                   'l3i': set(['23',])},
        'leaf': {str(k): range(26*k, 26*(k+1))
                 for k in range(24)}}

    test_tree = TaxonomyTree(
        data=full_tree_data)

    leaves_to_drop = ['23', '13', '14', '18']
    for leaf in leaves_to_drop:
        test_tree = test_tree.drop_node(level=test_tree.leaf_level, node=leaf)

    expected_tree_data = {
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': set(['l2b', 'l2d']),
                   'l1b': set(['l2a', 'l2e'])
                   },
        'level2': {'l2a': set(['l3b',]),
                   'l2b': set(['l3a', 'l3c']),
                   'l2d': set(['l3d', 'l3f', 'l3h']),
                   'l2e': set(['l3g',])},
        'level3': {'l3a': set([str(ii) for ii in range(3)]),
                   'l3b': set([str(ii) for ii in range(3, 7)]),
                   'l3c': set([str(ii) for ii in range(7, 9)]),
                   'l3d': set([str(ii) for ii in range(9, 13)]),
                   'l3f': set([str(ii) for ii in [15, 16, 17]]),
                   'l3g': set([str(ii) for ii in range(19, 21)]),
                   'l3h': set([str(ii) for ii in range(21, 23)])},
        'leaf': {str(k): range(26*k, 26*(k+1))
                 for k in [
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                     12, 15, 16, 17, 19, 20, 21, 22
                 ]}}
    expected_tree = TaxonomyTree(data=expected_tree_data)

    assert expected_tree == test_tree


def test_drop_node_errors():

    full_tree_data = {
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': set(['l2b', 'l2d']),
                   'l1b': set(['l2a', 'l2c', 'l2e']),
                   'l1c': set(['l2f',])
                   },
        'level2': {'l2a': set(['l3b',]),
                   'l2b': set(['l3a', 'l3c']),
                   'l2c': set(['l3e',]),
                   'l2d': set(['l3d', 'l3f', 'l3h']),
                   'l2e': set(['l3g',]),
                   'l2f': set(['l3i',])},
        'level3': {'l3a': set([str(ii) for ii in range(3)]),
                   'l3b': set([str(ii) for ii in range(3, 7)]),
                   'l3c': set([str(ii) for ii in range(7, 9)]),
                   'l3d': set([str(ii) for ii in range(9, 13)]),
                   'l3e': set([str(ii) for ii in range(13, 15)]),
                   'l3f': set([str(ii) for ii in range(15, 19)]),
                   'l3g': set([str(ii) for ii in range(19, 21)]),
                   'l3h': set([str(ii) for ii in range(21, 23)]),
                   'l3i': set(['23',])},
        'leaf': {str(k): range(26*k, 26*(k+1))
                 for k in range(24)}}

    tree = TaxonomyTree(data=full_tree_data)
    msg = "Level blah not present in tree"
    with pytest.raises(RuntimeError, match=msg):
        tree.drop_node(level='blah', node='garbage')
    msg = "Node garbage not present at level level2"
    with pytest.raises(RuntimeError, match=msg):
        tree.drop_node(level='level2', node='garbage')


def test_reverse_name_lookup():
    """
    Test TaxonomyTree method to lookup levels and nodes by
    human readable names.
    """
    data = {
        'hierarchy': ['A', 'B', 'C', 'D'],
        'A': {
            'a1': ['b1'],
            'a2': ['b2']
        },
        'B': {
            'b1': ['c1'],
            'b2': ['c2']
        },
        'C': {
            'c1': ['d1'],
            'c2': ['d2']
        },
        'D': {
            'd1': [],
            'd2': []
        },
        'hierarchy_mapper': {
            'A': 'levelA',
            'B': 'levelB',
            'C': 'levelC',
            'D': 'levelC'
        },
        'name_mapper': {
            'A': {
                'a1': {
                    'name': 'node_a1'
                },
                'a2': {
                    'name': 'node_a2'
                }
            },
            'B': {
                'b1': {
                    'name': 'node_b1'
                },
                'b2': {
                    'name': 'node_b1'
                }
            },
            'C': {
                'c1': {
                    'name': 'node_c1'
                },
                'c2': {
                    'name': 'node_c2'
                }
            },
            'D': {
                'd1': {
                    'name': 'node_d1'
                },
                'd2': {
                    'name': 'node_d2'
                }
            }
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        tree = TaxonomyTree(data=data)

    for level in ('A', 'B', 'C', 'D'):
        if level == 'D':
            expected_level = 'levelC'
        else:
            expected_level = f'level{level}'
        assert tree.level_to_name(level) == expected_level

        if expected_level == 'levelC':
            msg = 'levelC maps to many levels'
            with pytest.raises(RuntimeError, match=msg):
                tree.name_to_level(expected_level)
        else:
            assert tree.name_to_level(expected_level) == level

        for ii in (1, 2):
            node = f'{level.lower()}{ii}'
            if node == 'b2':
                expected_node = 'node_b1'
            else:
                expected_node = f'node_{node}'
            assert tree.label_to_name(
                level=level,
                label=node,
                name_key='name') == expected_node
            if expected_node == 'node_b1':
                msg = '(B, node_b1) maps to many nodes'
                with pytest.raises(RuntimeError, match=re.escape(msg)):
                    tree.name_to_node(level=level, node=expected_node)
            else:
                assert tree.name_to_node(
                    level=level,
                    node=expected_node) == (level, node)

            # test mapping from (readable_level, readable_node)
            if expected_level != 'levelC' and expected_node != 'node_b1':
                assert tree.name_to_node(
                    level=expected_level,
                    node=expected_node) == (level, node)

    msg = "E is not a valid level in this taxonomy"
    with pytest.raises(RuntimeError, match=msg):
        tree.name_to_level('E')
    with pytest.raises(RuntimeError, match=msg):
        tree.name_to_node(level='E', node='a2')

    msg = "(A, a3) not a valid node in this taxonomy"
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        tree.name_to_node(level='A', node='a3')

    assert tree.name_to_level('A') == 'A'
    assert tree.name_to_node(
        level='A',
        node='a2') == ('A', 'a2')

    assert tree.name_to_node(
        level='levelA',
        node='a2') == ('A', 'a2')


def test_parent_level():
    tree_data = {
        'hierarchy': ['level1', 'level2', 'level3', 'leaf'],
        'level1': {'l1a': set(['l2b', 'l2d']),
                   'l1b': set(['l2a', 'l2c', 'l2e']),
                   'l1c': set(['l2f',])
                   },
        'level2': {'l2a': set(['l3b',]),
                   'l2b': set(['l3a', 'l3c']),
                   'l2c': set(['l3e',]),
                   'l2d': set(['l3d', 'l3f', 'l3h']),
                   'l2e': set(['l3g',]),
                   'l2f': set(['l3i',])},
        'level3': {'l3a': set([str(ii) for ii in range(3)]),
                   'l3b': set([str(ii) for ii in range(3, 7)]),
                   'l3c': set([str(ii) for ii in range(7, 9)]),
                   'l3d': set([str(ii) for ii in range(9, 13)]),
                   'l3e': set([str(ii) for ii in range(13, 15)]),
                   'l3f': set([str(ii) for ii in range(15, 19)]),
                   'l3g': set([str(ii) for ii in range(19, 21)]),
                   'l3h': set([str(ii) for ii in range(21, 23)]),
                   'l3i': set(['23',])},
        'leaf': {str(k): range(26*k, 26*(k+1))
                 for k in range(24)}}

    tree = TaxonomyTree(data=tree_data)
    assert tree.parent_level('leaf') == 'level3'
    assert tree.parent_level('level3') == 'level2'
    assert tree.parent_level('level2') == 'level1'
    assert tree.parent_level('level1') is None
