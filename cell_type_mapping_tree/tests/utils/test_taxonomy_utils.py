import pytest
import copy
import numpy as np

from hierarchical_mapping.utils.taxonomy_utils import (
    get_taxonomy_tree,
    _get_rows_from_tree)


@pytest.fixture
def l1_to_l2_fixture():
    """
    Fixture modeling which level 2 objects belong
    to level 1
    """
    forward = {"l1a": set(["l2a", "l2d", "l2e"]),
               "l1b": set(["l2b", "l2f"]),
               "l1c": set(["l2c"])}

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward


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

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward

@pytest.fixture
def class_to_cluster_fixture(l2_to_class_fixture):
    """
    Fixture modeling which cluster objects belong
    to which class objects
    """
    list_of_classes = list(l2_to_class_fixture[1].keys())

    forward = dict()
    backward = dict()
    ct = 0
    for c in list_of_classes:
        forward[c] = set()
        for ii in range(4):
            this = f"clu_{ct}"
            ct += 1
            backward[this] = c
            forward[c].add(this)

    return forward, backward


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
        records_fixture):

    records = copy.deepcopy(records_fixture)

    column_hierarchy=["level1", "level2", "class", "cluster"]

    tree = get_taxonomy_tree(
                obs_records=records,
                column_hierarchy=column_hierarchy)

    assert tree['hierarchy'] == column_hierarchy
    assert tree["level1"] == l1_to_l2_fixture[0]
    assert tree["level2"] == l2_to_class_fixture[0]
    assert tree["class"] == class_to_cluster_fixture[0]

    row_set = set()
    for clu in tree["cluster"]:
        for ii in tree["cluster"][clu]:
            assert records[ii]["cluster"] == clu
            assert ii not in row_set
            row_set.add(ii)

    assert row_set == set(range(len(records)))



def test_get_rows_from_tree(
        records_fixture):

    row_to_cluster = dict()
    for ii, r in enumerate(records_fixture):
        row_to_cluster[ii] = r['cluster']

    column_hierarchy=["level1", "level2", "class", "cluster"]

    tree = get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)

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
