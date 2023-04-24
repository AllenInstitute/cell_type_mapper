import pytest

import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)

from hierarchical_mapping.marker_selection.marker_array import (
    MarkerGeneArray)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('marker_array'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_genes():
    return 14

@pytest.fixture
def n_cols():
    return 9

@pytest.fixture
def pair_to_idx_fixture():
    pair_to_idx = dict()
    pair_to_idx['level1'] = {
        'aa': {'bb': 0, 'cc': 1},
        'dd': {'ee': 2, 'ff': 3}}
    pair_to_idx['level2'] = {
        'a': {'b': 4, 'c': 5, 'd': 6},
        'e': {'f': 7, 'g': 8}}
    return pair_to_idx

@pytest.fixture
def is_marker_fixture(n_genes, n_cols):
    rng = np.random.default_rng(7401923)
    data = rng.integers(0, 2, (n_genes, n_cols), dtype=bool)
    return data

@pytest.fixture
def up_reg_fixture(n_genes, n_cols):
    rng = np.random.default_rng(1234567)
    data = rng.integers(0, 2, (n_genes, n_cols), dtype=bool)
    return data

@pytest.fixture
def gene_names_fixture(n_genes):
    return [f"g_{ii}" for ii in range(n_genes)]

@pytest.fixture
def backed_array_fixture(
        tmp_dir_fixture,
        is_marker_fixture,
        up_reg_fixture,
        pair_to_idx_fixture,
        gene_names_fixture,
        n_genes,
        n_cols):

    h5_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'))

    h5_path.unlink()

    marker = BinarizedBooleanArray(
        n_rows=n_genes,
        n_cols=n_cols)
    for i_row in range(n_genes):
        marker.set_row(i_row, is_marker_fixture[i_row, :])
    marker.write_to_h5(h5_path, h5_group='markers')

    up_reg = BinarizedBooleanArray(
        n_rows=n_genes,
        n_cols=n_cols)
    for i_row in range(n_genes):
        up_reg.set_row(i_row, up_reg_fixture[i_row, :])
    up_reg.write_to_h5(h5_path, h5_group='up_regulated')

    with h5py.File(h5_path, 'a') as dst:
        dst.create_dataset('n_pairs', data=n_cols)
        dst.create_dataset(
            'gene_names',
            data=json.dumps(gene_names_fixture).encode('utf-8'))
        dst.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_fixture).encode('utf-8'))

    return h5_path


def test_marker_gene_names(
        backed_array_fixture,
        gene_names_fixture,
        n_genes):

    arr = MarkerGeneArray(cache_path=backed_array_fixture)
    assert arr.gene_names == gene_names_fixture
    assert arr.n_genes == n_genes


def test_idx_of_pair_error(backed_array_fixture):
    arr = MarkerGeneArray(cache_path=backed_array_fixture)
    with pytest.raises(RuntimeError, match="not under taxonomy level level1"):
        arr.idx_of_pair(level='level1', node1='garbage', node2='other')

    with pytest.raises(RuntimeError, match="not a valid taxonomy pair"):
        arr.idx_of_pair(level='level1', node1='aa', node2='ff')

def test_idx_of_pair_idx(backed_array_fixture, pair_to_idx_fixture):
    arr = MarkerGeneArray(cache_path=backed_array_fixture)
    for level in pair_to_idx_fixture:
        for node1 in pair_to_idx_fixture[level]:
            for node2 in pair_to_idx_fixture[level][node1]:
                actual = arr.idx_of_pair(
                    level=level,
                    node1=node1,
                    node2=node2)
                expected = pair_to_idx_fixture[level][node1][node2]
                assert actual == expected

def test_marker_mask_from_gene_idx(
        backed_array_fixture,
        is_marker_fixture,
        up_reg_fixture,
        n_genes):
    arr = MarkerGeneArray(cache_path=backed_array_fixture)
    for i_gene in range(n_genes):
        (actual_marker,
         actual_up) = arr.marker_mask_from_gene_idx(gene_idx=i_gene)
        np.testing.assert_array_equal(
            actual_marker,
            is_marker_fixture[i_gene, :])
        np.testing.assert_array_equal(
            actual_up,
            up_reg_fixture[i_gene, :])


def test_marker_mask_from_pair_idx(
        backed_array_fixture,
        is_marker_fixture,
        up_reg_fixture,
        n_cols):
    arr = MarkerGeneArray(cache_path=backed_array_fixture)
    for i_col in range(n_cols):
        (actual_marker,
         actual_up) = arr.marker_mask_from_pair_idx(pair_idx=i_col)
        np.testing.assert_array_equal(
            actual_marker,
            is_marker_fixture[:, i_col])
        np.testing.assert_array_equal(
            actual_up,
            up_reg_fixture[:, i_col])
