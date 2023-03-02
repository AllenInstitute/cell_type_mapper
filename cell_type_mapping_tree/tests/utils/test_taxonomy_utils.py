import pytest
import numpy as np

from hierarchical_mapping.utils.taxonomy_utils import (
    get_taxonomy_tree)


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


def test_get_taxonomy_tree(
        l1_to_l2_fixture,
        l2_to_class_fixture,
        class_to_cluster_fixture):

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
