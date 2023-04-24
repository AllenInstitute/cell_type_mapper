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

from hierarchical_mapping.marker_selection.selection import (
    recalculate_utility_array)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('selection_module'))
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
    data = np.zeros((n_genes, n_cols), dtype=bool)
    data[2, 4] = True
    data[3, 4] = True
    data[11, 4] = True
    return data

@pytest.fixture
def up_reg_fixture(n_genes, n_cols):
    data = np.zeros((n_genes, n_cols), dtype=bool)
    data[2, 4] = True
    data[11, 4] = True
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


def test_recalculate_utilty_array(
        backed_array_fixture,
        n_genes):
    arr = MarkerGeneArray(cache_path=backed_array_fixture)
    util = np.zeros(n_genes, dtype=int)
    util = recalculate_utility_array(
        utility_array=util,
        marker_gene_array=arr,
        pair_idx=4,
        sign=-1)
    expected = np.zeros(n_genes, dtype=int)
    expected[3] = -1
    np.testing.assert_array_equal(expected, util)

    util = np.zeros(n_genes, dtype=int)
    util = recalculate_utility_array(
        utility_array=util,
        marker_gene_array=arr,
        pair_idx=4,
        sign=1)
    expected = np.zeros(n_genes, dtype=int)
    expected[2] = -1
    expected[11] = -1
    np.testing.assert_array_equal(expected, util)

    with pytest.raises(RuntimeError, match="Unclear how to interpret sign"):
        recalculate_utility_array(
            utility_array=util,
            marker_gene_array=arr,
            pair_idx=4,
            sign=3)
