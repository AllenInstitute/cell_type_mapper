import pytest

import h5py
import json
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)

from hierarchical_mapping.marker_selection.utils import (
    create_usefulness_array)


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

    with h5py.File(h5_path, "a") as out_file:
        out_file.create_dataset('n_pairs', data=n_cols)
        out_file.create_dataset(
            'gene_names',
            data=json.dumps(
                [f"g_{ii}" for ii in range(n_rows)]).encode('utf-8'))

    return h5_path


@pytest.mark.parametrize("gb_size", [1, 1.0e-7])
def test_create_usefulness_array(
        mask_array_fixture,
        backed_array_fixture,
        n_rows,
        gb_size):

    actual = create_usefulness_array(
        cache_path=backed_array_fixture,
        gb_size=gb_size)

    expected = np.sum(mask_array_fixture, axis=1)
    assert expected.shape == (n_rows, )
    np.testing.assert_array_equal(expected, actual)
