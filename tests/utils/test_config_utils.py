import pytest

import pathlib
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.config_utils import (
    patch_child_to_parent
)


@pytest.fixture(scope='module')
def child_to_parent_fixture(tmp_dir_fixture):

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture)
    )

    tmp_a = tempfile.mkdtemp(dir=tmp_dir)
    tmp_b = tempfile.mkdtemp(dir=tmp_dir)

    child_to_parent = dict()

    parent0 = mkstemp_clean(dir=tmp_a, prefix='parent0_', suffix='.txt')
    child0 = mkstemp_clean(dir=tmp_b, suffix='.txt')
    child_to_parent[child0] = parent0

    parent1 = mkstemp_clean(dir=tmp_a, prefix='parent1_', suffix='.txt')
    child1 = '/does/not/exist/child1.csv'
    child_to_parent[child1] = parent1

    parent2 = mkstemp_clean(dir=tmp_b, prefix='parent2_', suffix='.txt')
    child2 = '/really/does/not/exist/child2.csv'
    child_to_parent[child2] = parent2

    alt_path = pathlib.Path(parent2).parent / 'child2.csv'
    with open(alt_path, 'w') as dst:
        dst.write('here I am')

    yield child_to_parent

    _clean_up(tmp_dir)


@pytest.mark.parametrize('do_search', [True, False])
def test_patch_child_to_parent(
        child_to_parent_fixture,
        do_search):

    expected_missing_pairs = []
    expected_lookup = dict()
    if do_search:
        for k in child_to_parent_fixture:
            parent = pathlib.Path(child_to_parent_fixture[k])
            child = pathlib.Path(k)
            if parent.name.startswith('parent1_'):
                expected_missing_pairs.append(
                    (child_to_parent_fixture[k], k)
                )
            elif parent.name.startswith('parent0_'):
                expected_lookup[child] = parent
            else:
                alt = parent.parent / pathlib.Path(k).name
                expected_lookup[alt] = parent
    else:
        for k in child_to_parent_fixture:
            parent = pathlib.Path(child_to_parent_fixture[k])
            child = pathlib.Path(k)
            if parent.name.split('_')[0] in ('parent1', 'parent2'):
                expected_missing_pairs.append(
                    (child_to_parent_fixture[k], k)
                )
            else:
                expected_lookup[child] = parent

    (new_lookup,
     missing_pairs) = patch_child_to_parent(
         child_to_parent=child_to_parent_fixture,
         do_search=do_search)

    assert new_lookup == expected_lookup
    assert missing_pairs == expected_missing_pairs
    if do_search:
        assert len(missing_pairs) == 1
        assert len(new_lookup) == 2
    else:
        assert len(missing_pairs) == 2
        assert len(new_lookup) == 1
