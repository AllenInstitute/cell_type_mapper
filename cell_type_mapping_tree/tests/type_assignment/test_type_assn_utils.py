import h5py
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.type_assignment.utils import (
    reconcile_taxonomy_and_markers)


def test_taxonomy_reconciliation(tmp_path_factory):
    """
    Test method to validate taxonomy_tree against marker cache
    """
    tree_data = {
        'hierarchy': ['a', 'b', 'c'],
        'a': {
            'a1': ['b1', 'b2'],
            'a2': ['b3']
        },
        'b': {
            'b1': ['c1', 'c2', 'c3'],
            'b2': ['c4', 'c5'],
            'b3': ['c6', 'c7']
        },
        'c':{f'c{ii}': range(10*ii, 10*ii+5)
             for ii in (1, 2, 3, 4, 5, 6, 7)}
    }

    taxonomy_tree = TaxonomyTree(data=tree_data)

    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('assn_utils_'))

    # try valid case
    parent_list = ['None', 'a/a1', 'b/b1', 'b/b2', 'b/b3']
    # should not need 'a/a2' because it only has one child

    marker_path = mkstemp_clean(dir=tmp_dir, suffix='.h5')
    with h5py.File(marker_path, 'w') as out_file:
        for parent in parent_list:
            out_file.create_dataset(parent, data=b'abc')

    (flag, msg) = reconcile_taxonomy_and_markers(
        taxonomy_tree=taxonomy_tree,
        marker_cache_path=marker_path)
    assert flag
    assert len(msg) == 0

    # try case of missing parent
    parent_list = ['None', 'a/a1', 'b/b1', 'b/b3']
    # should not need 'a/a2' because it only has one child

    marker_path = mkstemp_clean(dir=tmp_dir, suffix='.h5')
    with h5py.File(marker_path, 'w') as out_file:
        for parent in parent_list:
            out_file.create_dataset(parent, data=b'abc')

    (flag, msg) = reconcile_taxonomy_and_markers(
        taxonomy_tree=taxonomy_tree,
        marker_cache_path=marker_path)

    assert not flag
    expected = "marker cache is missing parent 'b/b2'\n"
    assert msg == expected

    _clean_up(tmp_dir)
