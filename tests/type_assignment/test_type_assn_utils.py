import h5py
import numpy as np
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.type_assignment.utils import (
    reconcile_taxonomy_and_markers,
    infer_assignment)


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
        'c': {f'c{ii}': range(10*ii, 10*ii+5)
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

    # test case when all the parents at a given level are missing
    parent_list = ['None', 'a/a1']
    # should not need 'a/a2' because it only has one child

    marker_path = mkstemp_clean(dir=tmp_dir, suffix='.h5')
    with h5py.File(marker_path, 'w') as out_file:
        for parent in parent_list:
            out_file.create_dataset(parent, data=b'abc')

    (flag, msg) = reconcile_taxonomy_and_markers(
        taxonomy_tree=taxonomy_tree,
        marker_cache_path=marker_path)

    assert not flag
    expected = ("marker cache is missing all parents at level 'b'; "
                "consider running cell_type_mapper with "
                "--drop_level 'b'")
    assert msg == expected

    # test that it can handle missing root
    parent_list = ['a/a1', 'b/b1', 'b/b3']
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
    assert expected in msg
    expected = "marker cache is missing parent 'None'\n"
    assert expected in msg

    _clean_up(tmp_dir)


def test_infer_assignment():

    tree_data = {
        'hierarchy': ['level1', 'level2', 'level3'],
        'level1': {
            'a': ['c', 'd', 'e'],
            'b': ['f', 'g']
        },
        'level2': {
            'c': ['h', 'i'],
            'd': ['j', 'k'],
            'e': ['l', 'm'],
            'f': ['n', 'o'],
            'g': ['p']
        },
        'level3': {
            c: [] for c in 'hijklmnop'
        }
    }

    tree = TaxonomyTree(data=tree_data)

    votes = np.array(
        [[1, 3, 5, 0, 0, 0, 2],
         [0, 2, 4, 7, 0, 0, 1],
         [6, 0, 0, 1, 0, 0, 0]]
    )

    corr_sum = np.array(
        [[1.1, 2.2, 3.1, 0, 0, 0, 0.1],
         [0, 2.2, 4.4, 5.1, 0, 0, 6.7],
         [0.1, 0, 0, 0.9, 0, 0, 0]]
    )

    reference_types = np.array(
        ['p', 'h', 'n', 'j', 'k', 'o', 'i']
    )

    (new_votes,
     new_corr,
     new_reference_types) = infer_assignment(
         votes=votes,
         corr_sum=corr_sum,
         reference_types=reference_types,
         reference_level='level3',
         inference_level='level2',
         taxonomy_tree=tree
    )

    expected_types = np.array(['c', 'd', 'f', 'g'])
    expected_votes = np.array(
        [[5, 0, 5, 1],
         [3, 7, 4, 0],
         [0, 1, 0, 6]]
    )
    expected_corr = np.array(
      [[2.3, 0, 3.1, 1.1],
       [8.9, 5.1, 4.4, 0],
       [0, 0.9, 0, 0.1]]
    )

    np.testing.assert_array_equal(new_reference_types, expected_types)
    np.testing.assert_array_equal(new_votes, expected_votes)
    np.testing.assert_allclose(
        new_corr,
        expected_corr,
        atol=0.0,
        rtol=1.0e-6)

    (new_votes,
     new_corr,
     new_reference_types) = infer_assignment(
         votes=votes,
         corr_sum=corr_sum,
         reference_types=reference_types,
         reference_level='level3',
         inference_level='level1',
         taxonomy_tree=tree
    )

    expected_types = np.array(['a', 'b'])
    expected_votes = np.array(
        [[5, 6],
         [10, 4],
         [1, 6]]
    )
    expected_corr = np.array(
        [[2.3, 4.2],
         [14.0, 4.4],
         [0.9, 0.1]]
    )

    np.testing.assert_array_equal(new_reference_types, expected_types)
    np.testing.assert_array_equal(new_votes, expected_votes)
    np.testing.assert_allclose(
        new_corr,
        expected_corr,
        atol=0.0,
        rtol=1.0e-6)
