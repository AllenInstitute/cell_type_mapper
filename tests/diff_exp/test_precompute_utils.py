import pytest

import h5py
import json
import numpy as np

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_utils import (
    run_leaf_census)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('precompute_utils_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def taxonomy_tree_fixture():
    data = {
        'hierarchy': ['class', 'cluster'],
        'class': {
            'A': ['a', 'b', 'c', 'h', 'i', 'j'],
            'B': ['d', 'e', 'f', 'g']
        },
        'cluster': {
            n:[] for n in 'abcdefghij'
        }
    }
    tree = TaxonomyTree(data=data)
    return tree


@pytest.fixture(scope='module')
def expected_census(
        tmp_dir_fixture,
        taxonomy_tree_fixture):
    leaf_list = taxonomy_tree_fixture.all_leaves
    result = dict()
    for leaf in leaf_list:
        result[leaf] = dict()
    rng = np.random.default_rng(8712311)
    row_idx = list(range(len(leaf_list)))
    for i_file in range(3):
        pth = mkstemp_clean(
                dir=tmp_dir_fixture,
                prefix='for_census_',
                suffix='.h5')

        for leaf in leaf_list:
            result[leaf][str(pth)] = rng.integers(0, 255)
        rng.shuffle(row_idx)
        cluster_to_row = {
            leaf_list[ii]: row_idx[ii] for ii in range(len(leaf_list))}
        n_cells = np.zeros(len(leaf_list), dtype=int)
        for leaf in leaf_list:
            n_cells[cluster_to_row[leaf]] = int(result[leaf][str(pth)])
        with h5py.File(pth, 'w') as dst:
            dst.create_dataset(
                'taxonomy_tree',
                data=taxonomy_tree_fixture.to_str().encode('utf-8'))
            dst.create_dataset(
                'cluster_to_row',
                data=json.dumps(cluster_to_row).encode('utf-8'))
            dst.create_dataset(
                'n_cells',
                data=n_cells)

    return result


def test_leaf_census(
        expected_census,
        taxonomy_tree_fixture):
    precompute_path_list = list(expected_census['a'].keys())
    (actual_census,
     actual_tree) = run_leaf_census(precompute_path_list)
    assert actual_census == expected_census
    assert actual_tree.is_equal_to(taxonomy_tree_fixture)
