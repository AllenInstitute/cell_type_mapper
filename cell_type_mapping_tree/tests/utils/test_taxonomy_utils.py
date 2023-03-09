import pytest
import copy
import numpy as np
import json
import itertools

from hierarchical_mapping.utils.utils import json_clean_dict

from hierarchical_mapping.utils.taxonomy_utils import (
    get_taxonomy_tree,
    _get_rows_from_tree,
    compute_row_order,
    _get_leaves_from_tree,
    convert_tree_to_leaves,
    get_siblings,
    get_all_pairs)


@pytest.fixture
def column_hierarchy():
    return ["level1", "level2", "class", "cluster"]


@pytest.fixture
def l1_to_l2_fixture():
    """
    Fixture modeling which level 2 objects belong
    to level 1
    """
    forward = {"l1a": set(["l2a", "l2d", "l2e"]),
               "l1b": set(["l2b", "l2f"]),
               "l1c": set(["l2c"])}

    siblings = [
        ('level1', 'l1a', 'l1b'),
        ('level1', 'l1a', 'l1c'),
        ('level1', 'l1b', 'l1c'),
        ('level2', 'l2a', 'l2d'),
        ('level2', 'l2a', 'l2e'),
        ('level2', 'l2d', 'l2e'),
        ('level2', 'l2b', 'l2f')]

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward, siblings


@pytest.fixture
def l2_to_class_fixture():
    """
    Fixture modeling which class objects belong
    to which level 2 objects
    """
    forward = {"l2a": set(["c4", "c5"]),
               "l2b": set(["c1", "c6"]),
               "l2c": set(["c3"]),
               "l2d": set(["c2", "c7", "c8"]),
               "l2e": set(["c9"]),
               "l2f": set(["c10", "c11"])}

    siblings = [
        ('class', 'c4', 'c5'),
        ('class', 'c1', 'c6'),
        ('class', 'c2', 'c7'),
        ('class', 'c2', 'c8'),
        ('class', 'c7', 'c8'),
        ('class', 'c10', 'c11')]

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward, siblings

@pytest.fixture
def class_to_cluster_fixture(l2_to_class_fixture):
    """
    Fixture modeling which cluster objects belong
    to which class objects
    """
    rng = np.random.default_rng(98812)
    list_of_classes = list(l2_to_class_fixture[1].keys())

    forward = dict()
    backward = dict()
    siblings = []
    ct = 0
    for c in list_of_classes:
        forward[c] = set()
        children_list = []
        for ii in range(rng.integers(3, 7)):
            this = f"clu_{ct}"
            children_list.append(this)
            ct += 1
            backward[this] = c
            forward[c].add(this)
        children_list.sort()
        for pair in itertools.combinations(children_list, 2):
            siblings.append(('cluster', pair[0], pair[1]))

    return forward, backward, siblings


@pytest.fixture
def records_fixture(
         class_to_cluster_fixture,
         l2_to_class_fixture,
         l1_to_l2_fixture):
    rng = np.random.default_rng(871234)
    cluster_list = list(class_to_cluster_fixture[1].keys())
    records = []
    for ii in range(7):
        for clu in cluster_list:
            cl = class_to_cluster_fixture[1][clu]
            l2 = l2_to_class_fixture[1][cl]
            l1 = l1_to_l2_fixture[1][l2]
            this = {"cluster": clu,
                    "class": cl,
                    "level2": l2,
                    "level1": l1,
                    "garbage": rng.integers(8, 1000)}
            records.append(this)

    rng.shuffle(records)
    return records

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
    cleaned = json_clean_dict(result)
    json.dumps(cleaned)


@pytest.fixture
def parent_to_leaves(
        records_fixture,
        column_hierarchy):
    """
    Brute force your way to a dict mapping
    parent nodes to their ultimate leaves
    """
    # construct expected
    leaf_nodes = dict()
    parent_to_leaves = dict()
    leaf_class = column_hierarchy[-1]
    for record in records_fixture:
        this_leaf = record[leaf_class]
        leaf_nodes[this_leaf] = set([this_leaf])
        for h in column_hierarchy[:-1]:
            if h not in parent_to_leaves:
                parent_to_leaves[h] = dict()
            this_node = record[h]
            if this_node not in parent_to_leaves[h]:
                parent_to_leaves[h][this_node] = set()
            parent_to_leaves[h][this_node].add(record[leaf_class])

    parent_to_leaves[leaf_class] = leaf_nodes
    return parent_to_leaves

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
