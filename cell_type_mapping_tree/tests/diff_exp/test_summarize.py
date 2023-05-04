import pytest

import numpy as np

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)

from hierarchical_mapping.diff_exp.summarize import (
    can_we_summarize,
    summarize_from_arrays)


def test_can_we_summarize_no():

    rng = np.random.default_rng(88123)

    bad_gb = 0.0001
    n_rows =2**7
    n_cols = 1+np.ceil(bad_gb*8*1024**3/n_rows).astype(int)
    n_int = np.ceil(n_cols/8).astype(int)
    data = 255*np.ones((n_rows, n_int), dtype=np.uint8)
    marker_array = BinarizedBooleanArray.from_data_array(
        data_array=data,
        n_cols=n_cols)

    up_array = BinarizedBooleanArray.from_data_array(
        n_cols=n_cols,
        data_array=rng.integers(0, 255, (n_rows, n_int), dtype=np.uint8))

    actual = can_we_summarize(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=bad_gb)

    assert actual is None


@pytest.mark.parametrize(
    'expected_dtype', [np.uint8, np.uint16])
def test_can_we_summarize_yes(expected_dtype):

    rng = np.random.default_rng(22314)

    if expected_dtype == np.uint8:
        n_bits = 8
    else:
        n_bits = 16
    n_rows = 2**(n_bits-1)
    n_cols = 55

    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)

    marker_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    up_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    for i_row in range(n_rows):
        marker_array.set_row(i_row, marker_truth[i_row, :])
        up_array.set_row(i_row, up_truth[i_row, :])

    gb_good = 1.01*(16*marker_truth.sum()/(8*1024**3))

    actual = can_we_summarize(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=gb_good)

    assert actual is not None
    assert actual['gene_dtype'] == expected_dtype

    for i_col in range(n_cols):
        marker_col = marker_truth[:, i_col]
        up_col = up_truth[:, i_col]
        assert actual['up_col_sum'][i_col] == (marker_col*up_col).sum()
        assert actual['down_col_sum'][i_col] == (marker_col*(~up_col)).sum()


def test_summarize():
    n_bits = 8
    rng = np.random.default_rng(118231)
    n_rows = 2**(n_bits-1)
    n_cols = 112
    up_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth = rng.integers(0, 2, (n_rows, n_cols), dtype=bool)
    marker_truth[:, 17] = False

    up_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    marker_array = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    for i_row in range(n_rows):
        up_array.set_row(i_row, up_truth[i_row, :])
        marker_array.set_row(i_row, marker_truth[i_row, :])

    gb_estimate = 1.001*(n_bits*marker_truth.sum()/(8*1024**3))

    summary = summarize_from_arrays(
        marker_array=marker_array,
        up_array=up_array,
        gb_cutoff=gb_estimate)

    assert summary is not None

    up_v = summary['up_values']
    up_idx = summary['up_idx']
    down_v = summary['down_values']
    down_idx = summary['down_idx']

    for i_col in range(n_cols):
        marker_col = marker_truth[:, i_col]
        up_col = up_truth[:, i_col]

        up_values = np.where(
            np.logical_and(marker_col, up_col))[0]
        i0 = up_idx[i_col]
        if i_col < (n_cols+1):
            i1 = up_idx[i_col+1]
        else:
            i1 = n_cols
        actual = summary['up_values'][i0:i1]
        np.testing.assert_array_equal(actual, up_values)

        down_values = np.where(
            np.logical_and(marker_col, ~up_col))[0]
        i0 = down_idx[i_col]
        if i_col < (n_cols+1):
            i1 = down_idx[i_col+1]
        else:
            i1 = n_cols
        actual = summary['down_values'][i0:i1]
        np.testing.assert_array_equal(actual, down_values)
