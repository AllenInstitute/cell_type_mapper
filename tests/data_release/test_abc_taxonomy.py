"""
This module will test TaxonomyTree against the serialization scheme
adopted for the June 2023 ABC Atlas data release


probably actually want as input
cluster_to_cluster_annotation_membership.csv
    which will include labels and aliases
cluster_annotation_term.csv
    which encodes parent-child relationships
cell_metadata.csv
    which maps cells to clusters

apparently, alias is the only thing that's stable
aliases are unique within levels
but not across levels
"""
import pytest

import warnings

from cell_type_mapper.taxonomy.data_release_utils import (
    get_tree_above_leaves,
    get_label_to_name,
    get_cell_to_cluster_alias)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


def test_get_tree_above_leaves(
        cluster_annotation_term_fixture,
        cluster_to_supertype_fixture,
        supertype_to_subclass_fixture,
        subclass_to_class_fixture):

    actual = get_tree_above_leaves(
        csv_path=cluster_annotation_term_fixture,
        hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    assert len(actual) == 3
    assert 'class' in actual
    assert 'subclass' in actual
    assert 'supertype' in actual

    for lookup, parent_level in [(cluster_to_supertype_fixture, 'supertype'),
                                 (supertype_to_subclass_fixture, 'subclass'),
                                 (subclass_to_class_fixture, 'class')]:
        for child in lookup:
            parent = lookup[child]
            assert child in actual[parent_level][parent]


def test_get_label_to_name(
        cluster_membership_fixture,
        alias_fixture):

    actual = get_label_to_name(
        csv_path=cluster_membership_fixture,
        valid_term_set_labels=('cluster',))

    for full_label in alias_fixture:
        if 'cluster' in full_label:
            level = 'cluster'
        elif 'subclass' in full_label:
            continue
        elif 'supertype' in full_label:
            continue
        elif 'class' in full_label:
            continue
        else:
            raise RuntimeError(
                f"no obvious level for {full_label}")
        assert actual[(level, full_label)] == str(alias_fixture[full_label])


def test_full_label_to_name(
        cluster_membership_fixture,
        term_label_to_name_fixture):
    mapper = get_label_to_name(
        csv_path=cluster_membership_fixture,
        valid_term_set_labels=['class', 'subclass', 'supertype', 'cluster'],
        name_column='cluster_annotation_term_name')

    assert len(mapper) == len(term_label_to_name_fixture)
    assert mapper == term_label_to_name_fixture


def test_get_cell_to_cluster_alias(
        cell_metadata_fixture,
        alias_fixture,
        cell_to_cluster_fixture):

    actual = get_cell_to_cluster_alias(
        cell_metadata_path=cell_metadata_fixture)

    for cell in cell_to_cluster_fixture:
        assert (
            actual[cell]
            == str(alias_fixture[cell_to_cluster_fixture[cell]])
        )


def test_all_this(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        baseline_tree_fixture,
        baseline_tree_without_cells_fixture):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        test_tree = TaxonomyTree.from_data_release(
                cell_metadata_path=cell_metadata_fixture,
                cluster_annotation_path=cluster_annotation_term_fixture,
                cluster_membership_path=cluster_membership_fixture,
                hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    assert test_tree == baseline_tree_fixture
    assert test_tree != baseline_tree_without_cells_fixture


@pytest.mark.parametrize('do_pruning', [True, False])
def test_tree_from_incomplete_cell_metadata(
        incomplete_cell_metadata_fixture,
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        baseline_incomplete_tree_fixture,
        do_pruning):

    if not do_pruning:
        msg = "is not present in the keys at level cluster"
        with pytest.raises(RuntimeError, match=msg):
            TaxonomyTree.from_data_release(
                    cell_metadata_path=incomplete_cell_metadata_fixture,
                    cluster_annotation_path=cluster_annotation_term_fixture,
                    cluster_membership_path=cluster_membership_fixture,
                    hierarchy=['class', 'subclass', 'supertype', 'cluster'],
                    do_pruning=do_pruning)
    else:
        test_tree = TaxonomyTree.from_data_release(
                cell_metadata_path=incomplete_cell_metadata_fixture,
                cluster_annotation_path=cluster_annotation_term_fixture,
                cluster_membership_path=cluster_membership_fixture,
                hierarchy=['class', 'subclass', 'supertype', 'cluster'],
                do_pruning=do_pruning)

        assert test_tree == baseline_incomplete_tree_fixture


def test_no_cell_metadata(
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        baseline_tree_without_cells_fixture,
        baseline_tree_fixture):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        test_tree = TaxonomyTree.from_data_release(
                cell_metadata_path=None,
                cluster_annotation_path=cluster_annotation_term_fixture,
                cluster_membership_path=cluster_membership_fixture,
                hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    assert test_tree == baseline_tree_without_cells_fixture
    assert test_tree != baseline_tree_fixture


def test_de_aliasing(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        baseline_tree_fixture,
        alias_fixture,
        cell_to_cluster_fixture):

    test_tree = TaxonomyTree.from_data_release(
            cell_metadata_path=cell_metadata_fixture,
            cluster_annotation_path=cluster_annotation_term_fixture,
            cluster_membership_path=cluster_membership_fixture,
            hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    for cluster in set(cell_to_cluster_fixture.values()):
        alias = alias_fixture[cluster]
        assert test_tree.label_to_name(
            level='cluster',
            label=cluster,
            name_key='alias') == str(alias)


def test_name_mapping(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        baseline_tree_fixture,
        alias_fixture,
        cell_to_cluster_fixture,
        term_label_to_name_fixture):

    test_tree = TaxonomyTree.from_data_release(
            cell_metadata_path=cell_metadata_fixture,
            cluster_annotation_path=cluster_annotation_term_fixture,
            cluster_membership_path=cluster_membership_fixture,
            hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    for k in term_label_to_name_fixture:
        assert (
            test_tree.label_to_name(k[0], k[1])
            == term_label_to_name_fixture[k]
        )
    assert test_tree.label_to_name('junk', 'this_label') == 'this_label'
    assert test_tree.label_to_name('class', 'that_label') == 'that_label'

    other_data = {
        'hierarchy': ['a', 'b'],
        'a': {
            'c': ['d'], 'e': ['f']
        },
        'b': {
            'd': [], 'f': []
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        _ = TaxonomyTree(data=other_data)

    assert test_tree.label_to_name('a', 'x') == 'x'


def test_hierarchy_mapping(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        baseline_tree_fixture,
        alias_fixture,
        cell_to_cluster_fixture,
        term_label_to_name_fixture):

    test_tree = TaxonomyTree.from_data_release(
            cell_metadata_path=cell_metadata_fixture,
            cluster_annotation_path=cluster_annotation_term_fixture,
            cluster_membership_path=cluster_membership_fixture,
            hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    print(test_tree._data['hierarchy_mapper'])
    for level in ['class', 'subclass', 'supertype', 'cluster']:
        assert (
            test_tree.level_to_name(level_label=level)
            == f'{level}_readable'
        )
    assert test_tree.level_to_name(level_label='gar') == 'gar'


def test_abc_dropping(
        cell_metadata_fixture,
        cluster_membership_fixture,
        cluster_annotation_term_fixture,
        baseline_tree_fixture,
        alias_fixture,
        cell_to_cluster_fixture):
    """
    Just a smoke test; will check metadata, though
    """
    test_tree = TaxonomyTree.from_data_release(
            cell_metadata_path=cell_metadata_fixture,
            cluster_annotation_path=cluster_annotation_term_fixture,
            cluster_membership_path=cluster_membership_fixture,
            hierarchy=['class', 'subclass', 'supertype', 'cluster'])

    new_tree = test_tree.drop_level('supertype')
    assert new_tree._data['metadata']['dropped_levels'] == ['supertype']
    assert new_tree.hierarchy == ['class', 'subclass', 'cluster']
    new_tree = new_tree.drop_level('subclass')
    assert new_tree._data['metadata']['dropped_levels'] == ['supertype',
                                                            'subclass']
    assert new_tree.hierarchy == ['class', 'cluster']


def test_de_aliasing_when_no_map():
    data = {
        'hierarchy': ['a', 'b'],
        'a': {'aa': ['aaa'],
              'bb': ['bbb']},
        'b': {'aaa': ['1', '2'],
              'bbb': ['3']}}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        tree = TaxonomyTree(data=data)

    assert tree.label_to_name(level='cluster', label='3') == '3'
