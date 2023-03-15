import pytest

import h5py
import itertools
import json
import numpy as np
import os
import pathlib
import tempfile

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.taxonomy_utils import (
    get_all_leaf_pairs)

from hierarchical_mapping.marker_selection.utils import (
    _process_rank_chunk,
    select_marker_genes)



@pytest.mark.parametrize(
    "valid_rows, valid_genes, genes_per_pair, expected",
    [(set([2, 6, 7, 11]),
      set(np.arange(5, 27)),
      3,
      set([8, 22, 6, 9, 13])),
     (set([2, 6, 7, 11]),
      set(np.arange(5, 27)),
      2,
      set([8, 22, 6, 9])),
     (set([2, 6, 7, 11]),
      set([99, 98, 100]),
      3,
      set([])),
     (set([400, 401, 402]),
      set(np.arange(5, 27)),
      3,
      set([]))
    ])
def test_process_rank_chunk(
        valid_rows,
        valid_genes,
        genes_per_pair,
        expected):

    row0 = 4
    row1 = 15
    rank_chunk = np.ones((row1-row0, 14), dtype=int)
    rank_chunk[2][2] = 6
    rank_chunk[2][7] = 9
    rank_chunk[2][8] = 13
    rank_chunk[2][10] = 17

    # this row should not be valid
    rank_chunk[4][1] = 18
    rank_chunk[4][2] = 19
    rank_chunk[4][3] = 20

    rank_chunk[7][11] = 8
    rank_chunk[7][12] = 22

    actual = _process_rank_chunk(
                valid_rows=valid_rows,
                valid_genes=valid_genes,
                rank_chunk=rank_chunk,
                row0=row0,
                row1=row1,
                genes_per_pair=genes_per_pair)

    assert actual == expected



@pytest.fixture
def tree_fixture():
    rng = np.random.default_rng(712312)
    taxonomy_tree = dict()
    hierarchy = ['class', 'subclass', 'division', 'cluster']
    available = dict()
    uuid = 0
    for i_parent in range(len(hierarchy)-1):
        parent_level = hierarchy[i_parent]
        child_level = hierarchy[i_parent+1]
        if parent_level not in available:
            parent_list = [f'{parent_level}_0', f'{parent_level}_1']
        else:
            parent_list = available[parent_level]
        available[child_level] = []
        taxonomy_tree[parent_level] = dict()
        for parent in parent_list:
            n_children = rng.integers(2,5)
            taxonomy_tree[parent_level][parent] = []
            for ii in range(n_children):
                name = f'{child_level}_{uuid}'
                uuid += 1
                taxonomy_tree[parent_level][parent].append(name)
                available[child_level].append(name)

    leaf_level = hierarchy[-1]
    taxonomy_tree[leaf_level] = dict()
    for ii, leaf in enumerate(available[leaf_level]):
        taxonomy_tree[leaf_level][leaf] = [2*ii, 2*ii+1]

    taxonomy_tree['hierarchy'] = hierarchy
    return taxonomy_tree


@pytest.fixture
def gene_names_fixture():
    result = []
    for ii in range(47):
        result.append(f'g_{ii}')
    return result


@pytest.fixture
def score_path_fixture(
        tree_fixture,
        gene_names_fixture,
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('marker_selection'))

    rng = np.random.default_rng(667123)

    score_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5')
    os.close(score_path[0])
    score_path = pathlib.Path(score_path[1])

    parent = tree_fixture['hierarchy'][-2]
    leaves = []
    for k in tree_fixture[parent]:
        leaves += list(tree_fixture[parent][k])
    pair_to_idx = dict()
    leaf_level = tree_fixture['hierarchy'][-1]
    pair_to_idx[leaf_level] = dict()
    idx = 0
    for pair in itertools.combinations(leaves, 2):
        if pair[0] not in pair_to_idx[leaf_level]:
            pair_to_idx[leaf_level][pair[0]] = dict()
        if pair[1] not in pair_to_idx[leaf_level]:
            pair_to_idx[leaf_level][pair[1]] = dict()
        pair_to_idx[leaf_level][pair[1]][pair[0]] = idx
        pair_to_idx[leaf_level][pair[0]][pair[1]] = idx
        idx += 1

    data = rng.integers(0,
                        len(gene_names_fixture),
                        size=(idx, 7)).astype(np.uint8)

    with h5py.File(score_path, 'w') as out_file:
        out_file.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        out_file.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx).encode('utf-8'))
        out_file.create_dataset(
            'ranked_list',
            data=data,
            chunks=(len(gene_names_fixture)//4, 7),
            compression='gzip')

    yield score_path

    _clean_up(tmp_dir)


@pytest.mark.parametrize("null_parent_node", (True, False))
def test_select_marker_genes_multiprocessing(
        tree_fixture,
        score_path_fixture,
        gene_names_fixture,
        null_parent_node):
    """
    This test assumes that running select_marker_genes with
    n_processors=1 gives the right answer. The point of this
    test is to make sure that the result aggregation logic
    for n_processors > 1 runs correctly.
    """

    rng = np.random.default_rng(776123)
    query_genes = rng.choice(gene_names_fixture,
                             len(gene_names_fixture)//3,
                             replace=False)

    if null_parent_node:
        parent_node = None
    else:
        level = tree_fixture['hierarchy'][1]
        k_list = list(tree_fixture[level].keys())
        parent_node = (level, k_list[1])

    leaf_pair_list = get_all_leaf_pairs(
        taxonomy_tree=tree_fixture,
        parent_node=parent_node)

    baseline = select_marker_genes(
        score_path=score_path_fixture,
        leaf_pair_list=leaf_pair_list,
        query_genes=query_genes,
        genes_per_pair=2,
        n_processors=1,
        rows_at_a_time=1000000)

    assert len(baseline['reference']) > 0

    test = select_marker_genes(
        score_path=score_path_fixture,
        leaf_pair_list=leaf_pair_list,
        query_genes=query_genes,
        genes_per_pair=2,
        n_processors=3,
        rows_at_a_time=17)

    assert test == baseline

    for lookup in (test, baseline):
        assert len(lookup['reference']) == len(lookup['query'])
        for ii in range(len(lookup['reference'])):
            rr = lookup['reference'][ii]
            qq = lookup['query'][ii]
            assert gene_names_fixture[rr] == query_genes[qq]
