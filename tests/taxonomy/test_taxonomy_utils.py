import pytest

import anndata
import copy
import json
import itertools
import pandas as pd
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.utils.utils import clean_for_json

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree,
    prune_by_h5ad
)

from cell_type_mapper.taxonomy.utils import (
    get_taxonomy_tree,
    _get_rows_from_tree,
    compute_row_order,
    _get_leaves_from_tree,
    convert_tree_to_leaves,
    get_siblings,
    get_all_pairs,
    get_all_leaf_pairs,
    validate_taxonomy_tree,
    get_child_to_parent,
    prune_tree)


def test_get_taxonomy_tree(
        l1_to_l2_fixture,
        l2_to_class_fixture,
        class_to_cluster_fixture,
        records_fixture,
        column_hierarchy):

    input_records = copy.deepcopy(records_fixture)

    tree = get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

    # make sure input did not change
    assert input_records == records_fixture

    assert tree['hierarchy'] == column_hierarchy
    assert tree["level1"] == l1_to_l2_fixture[0]
    assert tree["level2"] == l2_to_class_fixture[0]
    assert tree["class"] == class_to_cluster_fixture[0]

    row_set = set()
    for clu in tree["cluster"]:
        for ii in tree["cluster"][clu]:
            assert records_fixture[ii]["cluster"] == clu
            assert ii not in row_set
            row_set.add(ii)

    assert row_set == set(range(len(records_fixture)))


def test_get_rows_from_tree(
        records_fixture,
        column_hierarchy):

    input_records = copy.deepcopy(records_fixture)

    row_to_cluster = dict()
    for ii, r in enumerate(records_fixture):
        row_to_cluster[ii] = r['cluster']

    tree = get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

    # make sure input did not change
    assert input_records == records_fixture

    for level in column_hierarchy[:-1]:
        for this in tree[level]:

            # which rows should be returned, regardless
            # of order
            expected = set()
            for ii, record in enumerate(records_fixture):
                if record[level] == this:
                    expected.add(ii)

            actual = _get_rows_from_tree(
                        tree=tree,
                        level=level,
                        this_node=this)

            # check that rows are returned in blocks defined
            # by the leaf node
            current_cluster = None
            checked_clusters = set()
            for ii in actual:
                this_cluster = row_to_cluster[ii]
                if current_cluster is None:
                    current_cluster = this_cluster
                    checked_clusters.add(this_cluster)
                else:
                    if this_cluster != current_cluster:
                        # make sure we have not backtracked to a previous
                        # cluster
                        assert this_cluster not in checked_clusters
                        checked_clusters.add(this_cluster)
                        current_cluster = this_cluster

            assert len(checked_clusters) > 0
            assert len(checked_clusters) < len(actual)
            assert len(actual) == len(expected)
            assert set(actual) == expected


def test_compute_row_order(
        records_fixture,
        column_hierarchy):

    input_records = copy.deepcopy(records_fixture)

    result = compute_row_order(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

    # make sure input didn't change in place
    assert input_records == records_fixture

    current = dict()
    checked = dict()
    for c in column_hierarchy:
        checked[c] = set()
        current[c] = None

    # make sure rows are contiguous
    for ii in result["row_order"]:
        this_record = records_fixture[ii]
        for c in column_hierarchy:
            if current[c] is None:
                current[c] = this_record[c]
                checked[c].add(this_record[c])
            else:
                if this_record[c] != current[c]:
                    # make sure we haven't backtracked
                    assert this_record[c] not in checked[c]
                    checked[c].add(this_record[c])
                    current[c] = this_record[c]

    assert result["tree"]["hierarchy"] == column_hierarchy

    orig_tree = get_taxonomy_tree(
                    obs_records=records_fixture,
                    column_hierarchy=column_hierarchy)

    for c in column_hierarchy[:-1]:
        assert result["tree"][c] == orig_tree[c]

    # check remapped_rows
    leaf_class = column_hierarchy[-1]
    assert result["tree"][leaf_class] != orig_tree[leaf_class]
    for node in orig_tree[leaf_class]:
        orig_rows = orig_tree[leaf_class][node]
        remapped_rows = [result["row_order"][n]
                         for n in result["tree"][leaf_class][node]]
        assert remapped_rows == orig_rows


def test_cleaning(
        records_fixture,
        column_hierarchy):

    result = compute_row_order(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)
    cleaned = clean_for_json(result)
    json.dumps(cleaned)


def test_leaves_from_tree(
        records_fixture,
        column_hierarchy,
        parent_to_leaves):

    tree = get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

    for h in column_hierarchy:
        for node in parent_to_leaves[h]:
            expected = parent_to_leaves[h][node]
            actual = _get_leaves_from_tree(
                         tree=tree,
                         level=h,
                         this_node=node)
            assert set(actual) == expected
            assert len(set(actual)) == len(actual)


def test_convert_tree_to_leaves(
        records_fixture,
        column_hierarchy,
        parent_to_leaves):

    tree = get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

    as_leaves = convert_tree_to_leaves(
                    taxonomy_tree=tree)

    for h in column_hierarchy:
        for this_node in parent_to_leaves[h]:
            expected = parent_to_leaves[h][this_node]
            actual = as_leaves[h][this_node]
            assert set(actual) == expected
            assert len(set(actual)) == len(actual)


def test_get_siblings(
        records_fixture,
        column_hierarchy,
        l1_to_l2_fixture,
        l2_to_class_fixture,
        class_to_cluster_fixture):

    tree = get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

    siblings = get_siblings(tree)

    ct = 0
    for expected in (l1_to_l2_fixture[2],
                     l2_to_class_fixture[2],
                     class_to_cluster_fixture[2]):
        for el in expected:
            assert el in siblings
            ct += 1
    assert len(siblings) == ct


def test_get_all_pairs(
        records_fixture,
        column_hierarchy,
        l1_to_l2_fixture,
        l2_to_class_fixture,
        class_to_cluster_fixture):

    tree = get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

    siblings = get_all_pairs(tree)
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


def test_get_all_leaf_pairs():

    tree = {
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
        'leaf': {str(k): range(k, 26*k, 26*(k+1))
                 for k in range(24)}}

    # check non-None parent Node
    parent_node = ('level2', 'l2b')
    actual = get_all_leaf_pairs(
                taxonomy_tree=tree,
                parent_node=parent_node)

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

    actual = get_all_leaf_pairs(
                taxonomy_tree=tree,
                parent_node=parent_node)

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
    actual = get_all_leaf_pairs(
                taxonomy_tree=tree,
                parent_node=parent_node)
    assert actual == []

    actual = get_all_leaf_pairs(
                taxonomy_tree=tree,
                parent_node=('leaf', '15'))
    assert actual == []


def test_get_taxonomy_tree_errors():
    """
    Test that, in case where a child appears to have multiple parents
    in a taxonomy, get_taxonomy_tree throws an error.
    """
    obs_records = [
        {'l1': 'a', 'l2': 'bb', 'l3': 'ccc'},
        {'l1': 'a', 'l2': 'aa', 'l3': 'ddd'},
        {'l1': 'b', 'l2': 'cc', 'l3': 'ccc'}
    ]

    with pytest.raises(RuntimeError, match='ccc has at least two parents'):
        get_taxonomy_tree(
            obs_records=obs_records,
            column_hierarchy=['l1', 'l2', 'l3'])


def test_validate_taxonomy_tree():

    # missing taxons
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': ['aaa', 'bbb'],
            'bb': ['ccc', 'ddd']
        },
        'b': {
            'aaa': [1, 2, 3, 4]
        }
    }

    with pytest.raises(RuntimeError,
                       match="not present in the keys"):
        validate_taxonomy_tree(tree)

    # multiple parents
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': ['aaa', 'bbb'],
            'bb': ['bbb']
        },
        'b': {
            'aaa': [1, 2, 3, 4],
            'bbb': [7, 8, 9]
        }
    }

    with pytest.raises(RuntimeError,
                       match="at least two parents"):
        validate_taxonomy_tree(tree)

    # extra key
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': ['aaa', 'bbb'],
            'bb': ['ccc']
        },
        'b': {
            'aaa': [1, 2, 3, 4],
            'bbb': [7, 8, 9],
            'ccc': [11, 12, 13]
        },
        'c': 'hello'
    }

    with pytest.raises(RuntimeError,
                       match="Expect tree to have keys"):
        validate_taxonomy_tree(tree)

    # missing 'hierarchy' key
    tree = {
        'a': {
            'aa': ['aaa', 'bbb'],
            'bb': ['ccc']
        },
        'b': {
            'aaa': [1, 2, 3, 4],
            'bbb': [7, 8, 9],
            'ccc': [11, 12, 13]
        }
    }

    with pytest.raises(RuntimeError,
                       match="tree has no 'hierarchy'"):
        validate_taxonomy_tree(tree)

    # missing level key
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': ['aaa', 'bbb'],
            'bb': ['ccc']
        }
    }

    with pytest.raises(RuntimeError,
                       match="Expect tree to have keys"):
        validate_taxonomy_tree(tree)

    # case of missing child
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': ['1', '2'],
            'bb': ['3', '4']
        },
        'b': {
            '1': [1, 2, 3, 4],
            '2': [5, 6, 7],
            '3': [8, 9, 10]
        }
    }

    with pytest.raises(RuntimeError,
                       match="not present in the keys"):
        validate_taxonomy_tree(tree)

    # case of missing parent
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': ['1'],
            'bb': ['3']
        },
        'b': {
            '1': [1, 2, 3, 4],
            '2': [5, 6, 7],
            '3': [8, 9, 10]
        }
    }

    with pytest.raises(RuntimeError,
                       match="has no parent"):
        validate_taxonomy_tree(tree)

    # keys that are not strings
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': [1, 2],
            'bb': [3]
        },
        'b': {
            1: [1, 2, 3, 4],
            2: [5, 6, 7],
            3: [8, 9, 10]
        }
    }

    with pytest.raises(RuntimeError,
                       match="not a str"):
        validate_taxonomy_tree(tree)

    # repeated rows
    tree = {
        'hierarchy': ['a', 'b'],
        'a': {
            'aa': ['1', '2'],
            'bb': ['3']
        },
        'b': {
            '1': [1, 2, 3, 4],
            '2': [5, 6, 7, 4],
            '3': [8, 9, 10]
        }
    }

    with pytest.raises(RuntimeError,
                       match="Some rows appear more than once"):
        validate_taxonomy_tree(tree)


def test_get_child_to_parent():
    """
    Test the child-to-parent utility function
    """
    tree = {
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

        validate_taxonomy_tree(tree)

    actual = get_child_to_parent(tree_data=tree)

    expected = {
        'c': {
            'u': '3',
            'v': '3',
            'w': '3',
            'x': '1',
            'y': '2',
            'z': '2'
        },
        'b': {
            '1': 'aa',
            '2': 'aa',
            '3': 'bb'
        }
    }

    assert actual == expected


def test_prune_tree():

    tree = {
        'hierarchy': ['a', 'b', 'c'],
        'a': {
            'aa': ['1', '2'],
            'bb': ['3'],
            'cc': ['4'],
            'dd': ['6', '7', '8'],
            'ee': ['9']
        },
        'b': {
            '1': ['x'],
            '2': ['y', 'z'],
            '3': ['w', 'u', 'v'],
            '5': ['t'],
            '7': ['s'],
            '8': ['r', 'p'],
            '9': ['o', 'n', 'm']
        },
        'c': {
            'r': [],
            'u': [],
            'v': [],
            'w': [],
            'x': [],
            'y': [],
            'z': []
        }
    }

    expected = {
        'hierarchy': ['a', 'b', 'c'],
        'a': {
            'aa': ['1', '2'],
            'bb': ['3'],
            'dd': ['8']
        },
        'b': {
            '1': ['x'],
            '2': ['y', 'z'],
            '3': ['w', 'u', 'v'],
            '8': ['r']
        },
        'c': {
            'r': [],
            'u': [],
            'v': [],
            'w': [],
            'x': [],
            'y': [],
            'z': []
        }
    }

    tree = prune_tree(tree)
    assert tree == expected


def test_prune_by_h5ad(tmp_dir_fixture):

    data = {
        'hierarchy': ['l0', 'l1', 'l2'],
        'l0': {
           'A': ['a', 'b'],
           'B': ['c'],
           'C': ['d', 'e']
        },
        'l1': {
           'a': ['aa'],
           'b': ['bb', 'cc'],
           'c': ['dd', 'ee'],
           'd': ['ff'],
           'e': ['gg', 'hh']
        },
        'l2': {
            'aa': ['c0', 'c1'],
            'bb': ['c2', 'c3'],
            'cc': ['c4', 'c5'],
            'dd': ['c6', 'c7'],
            'ee': ['c8', 'c9'],
            'ff': ['c10', 'c11'],
            'gg': ['c12', 'c13'],
            'hh': ['c14', 'c15']
        }
    }

    initial_tree = TaxonomyTree(data=data)

    h5ad_path_list = []
    for i0, i1, in [(0, 7), (7, 12), (12, 16)]:
        pth = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'
        )
        obs = pd.DataFrame(
            [{'cell': f'c{ii}'} for ii in range(i0, i1, 2)]
        ).set_index('cell')
        aa = anndata.AnnData(obs=obs)
        aa.write_h5ad(pth)
        h5ad_path_list.append(pth)

    full_tree = prune_by_h5ad(
        taxonomy_tree=initial_tree,
        h5ad_list=h5ad_path_list)

    assert full_tree == initial_tree

    abbreviated_tree = prune_by_h5ad(
        taxonomy_tree=initial_tree,
        h5ad_list=h5ad_path_list[:-1])

    assert abbreviated_tree != initial_tree

    abbr_data = {
        'hierarchy': ['l0', 'l1', 'l2'],
        'l0': {
           'A': ['a', 'b'],
           'B': ['c'],
           'C': ['d']
        },
        'l1': {
           'a': ['aa'],
           'b': ['bb', 'cc'],
           'c': ['dd', 'ee'],
           'd': ['ff'],
        },
        'l2': {
            'aa': [],
            'bb': [],
            'cc': [],
            'dd': [],
            'ee': [],
            'ff': []
        }
    }

    expected_tree = TaxonomyTree(abbr_data)
    assert abbreviated_tree.is_equal_to(expected_tree)
