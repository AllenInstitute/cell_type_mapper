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
    rng =np.random.default_rng(554422)
    data = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    return data

@pytest.fixture
def up_regulated_fixture(
        n_rows,
        n_cols):
    rng =np.random.default_rng(118823)
    data = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    return data


@pytest.fixture
def backed_array_fixture(
        mask_array_fixture,
        up_regulated_fixture,
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
        arr.set_col(i_col, up_regulated_fixture[:, i_col])

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


def expected_utility_score(
        mask_array,
        up_regulated,
        taxonomy_idx,
        census):

    if taxonomy_idx is not None:
        mask_array = mask_array[:, taxonomy_idx]
        up_regulated = up_regulated[:, taxonomy_idx]

    unq_census_values, unq_ct = np.unique(census, return_counts=True)
    np.testing.assert_array_equal(unq_census_values, np.sort(unq_census_values))
    value_to_score = dict()
    value_to_score[0] = 0.0

    max_log_bonus = np.log10((census.shape[0]*census.shape[1])**2)

    for ii, val in enumerate(unq_census_values):
        if val == 0:
            continue
        this_score = len(unq_census_values) - ii
        ct = unq_ct[:ii].sum()
        n_log = ct//(census.size//10)
        bonus = np.power(10, max_log_bonus-2*n_log)
        value_to_score[val] = this_score + bonus

    scores = np.zeros(census.shape, dtype=float)
    for i_row in range(census.shape[0]):
        for i_col in range(census.shape[1]):
            scores[i_row, i_col] = value_to_score[census[i_row, i_col]]

    utility = np.zeros(mask_array.shape[0], dtype=float)
    for i_row in range(mask_array.shape[0]):
        for i_col in range(mask_array.shape[1]):
            if mask_array[i_row, i_col]:
                if up_regulated[i_row, i_col]:
                    utility[i_row] += value_to_score[census[i_col, 1]]
                else:
                    utility[i_row] += value_to_score[census[i_col, 0]]
    return utility, scores

@pytest.mark.parametrize(
        "gb_size, taxonomy_mask",
        product([1, 1.0e-7],
                [None, np.array([13, 22, 81, 37])]))
def test_create_utility_array(
        mask_array_fixture,
        up_regulated_fixture,
        backed_array_fixture,
        n_rows,
        n_cols,
        gb_size,
        taxonomy_mask):

    arr = MarkerGeneArray(cache_path=backed_array_fixture)

    (actual_util,
     actual_census,
     actual_scores) = create_utility_array(
        marker_gene_array=arr,
        gb_size=gb_size,
        taxonomy_mask=taxonomy_mask)

    if taxonomy_mask is None:
        expected_census = np.zeros((mask_array_fixture.shape[1], 2), dtype=int)
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                if mask_array_fixture[i_row, i_col]:
                    if up_regulated_fixture[i_row, i_col]:
                        expected_census[i_col, 1] += 1
                    else:
                        expected_census[i_col, 0] += 1
    else:
        expected_census = np.zeros((len(taxonomy_mask), 2), dtype=int)
        for i_row in range(n_rows):
            for i_taxon, i_col in enumerate(taxonomy_mask):
                if mask_array_fixture[i_row, i_col]:
                    if up_regulated_fixture[i_row, i_col]:
                        expected_census[i_taxon, 1] += 1
                    else:
                        expected_census[i_taxon, 0] += 1

    (expected_util,
     expected_scores) = expected_utility_score(
        mask_array=mask_array_fixture,
        up_regulated=up_regulated_fixture,
        taxonomy_idx=taxonomy_mask,
        census=expected_census)

    np.testing.assert_array_equal(expected_census, actual_census)
    assert expected_census.sum() > 0

    assert expected_util.shape == (n_rows, )
    np.testing.assert_allclose(expected_util,
                               actual_util,
                               atol=0.0,
                               rtol=1.0e-6)
    assert expected_util.sum() > 0

    np.testing.assert_allclose(
        expected_scores,
        actual_scores,
        atol=0.0,
        rtol=1.0e-7)
