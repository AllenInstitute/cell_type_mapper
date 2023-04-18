import pytest

import numpy as np

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray)


@pytest.mark.parametrize(
        "n_rows, n_cols", [(9, 13), (64, 128), (57, 77)])
def test_binarized_array_roundtrip(
        n_rows,
        n_cols):
    rng = np.random.default_rng(66771223)
    arr = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    src = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)
    for i_row in range(n_rows):
        arr.set_row(i_row, src[i_row, :])

    for i_row in range(n_rows):
        actual = arr.get_row(i_row)
        np.testing.assert_array_equal(actual, src[i_row, :])

    for i_col in range(n_cols):
        actual = arr.get_col(i_col)
        np.testing.assert_array_equal(actual, src[:, i_col])
