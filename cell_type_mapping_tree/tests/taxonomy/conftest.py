import pytest

import itertools
import numpy as np


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
