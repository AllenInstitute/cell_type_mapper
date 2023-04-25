import pytest

import h5py
from itertools import product
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)

from hierarchical_mapping.marker_selection.marker_array import (
    MarkerGeneArray)

from hierarchical_mapping.marker_selection.utils import (
    create_utility_array)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('marker_array'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_rows():
    return 551


@pytest.fixture
def n_cols():
    return 245

@pytest.fixture
def mask_array_fixture(
        n_rows,
        n_cols):
    rng =np.random.default_rng()
    data = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    return data

@pytest.fixture
def backed_array_fixture(
        mask_array_fixture,
        tmp_dir_fixture,
        n_rows,
        n_cols):

    h5_path = pathlib.Path(
        mkstemp_clean(dir=tmp_dir_fixture,
                      suffix='.h5'))
    h5_path.unlink()

    arr = BackedBinarizedBooleanArray(
        h5_path=h5_path,
        h5_group='markers',
        n_rows=n_rows,
        n_cols=n_cols,
        read_only=False)

    for i_col in range(n_cols):
        arr.set_col(i_col, mask_array_fixture[:, i_col])

    del arr

    arr = BackedBinarizedBooleanArray(
        h5_path=h5_path,
        h5_group='up_regulated',
        n_rows=n_rows,
        n_cols=n_cols,
        read_only=False)

    for i_col in range(n_cols):
        arr.set_col(i_col, mask_array_fixture[:, i_col])

    del arr


    with h5py.File(h5_path, "a") as out_file:
        out_file.create_dataset('n_pairs', data=n_cols)
        out_file.create_dataset(
            'gene_names',
            data=json.dumps(
                [f"g_{ii}" for ii in range(n_rows)]).encode('utf-8'))
        out_file.create_dataset(
            'pair_to_idx',
            data=json.dumps({'a': 1, 'b':2}).encode('utf-8'))

    return h5_path


@pytest.mark.parametrize(
        "gb_size, taxonomy_mask",
        product([1, 1.0e-7],
                [None, np.array([13, 22, 81, 37])]))
def test_create_utility_array(
        mask_array_fixture,
        backed_array_fixture,
        n_rows,
        gb_size,
        taxonomy_mask):

    arr = MarkerGeneArray(cache_path=backed_array_fixture)

    (actual,
     actual_census) = create_utility_array(
        marker_gene_array=arr,
        gb_size=gb_size,
        taxonomy_mask=taxonomy_mask)

    if taxonomy_mask is None:
        expected = np.sum(mask_array_fixture, axis=1)
        expected_census = np.sum(mask_array_fixture, axis=0)
    else:
        expected = np.zeros(n_rows, dtype=int)
        expected_census = np.zeros(len(taxonomy_mask), dtype=int)
        for i_row in range(n_rows):
            for i_taxon, i_col in enumerate(taxonomy_mask):
                if mask_array_fixture[i_row, i_col]:
                    expected[i_row] += 1
                    expected_census[i_taxon] += 1
    assert expected.shape == (n_rows, )
    np.testing.assert_array_equal(expected, actual)
    assert expected.sum() > 0
    np.testing.assert_array_equal(expected_census, actual_census)
    assert expected_census.sum() > 0
