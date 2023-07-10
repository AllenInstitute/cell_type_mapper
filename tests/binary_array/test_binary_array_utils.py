import pytest

from itertools import product
import numpy as np

from cell_type_mapper.binary_array.utils import (
    binarize_boolean_array,
    unpack_binarized_boolean_array,
    unpack_binarized_boolean_array_2D)



@pytest.mark.parametrize(
        "n_bool", [8, 32, 64, 57, 13, 23, 5])
def test_binary_utils_roundtrip(n_bool):
    rng = np.random.default_rng(6677122)
    src = rng.integers(0, 2, n_bool).astype(bool)
    binarized = binarize_boolean_array(src)
    actual = unpack_binarized_boolean_array(
        binarized_data=binarized,
        n_booleans=n_bool)
    np.testing.assert_array_equal(src, actual)
    assert binarized.dtype == np.uint8
    assert len(binarized) == np.ceil(n_bool/8).astype(int)


@pytest.mark.parametrize(
        "n_cols, n_rows", product(range(24, 33), [13, 22, 7]))
def test_unpack_binarized_array_2D(n_cols, n_rows):
    rng = np.random.default_rng(556123)
    n_ints = np.ceil(n_cols/8).astype(int)
    data = rng.integers(0, 255, (n_rows, n_ints), dtype=np.uint8)
    actual = unpack_binarized_boolean_array_2D(
        binarized_data=data,
        n_booleans=n_cols)
    for i_row in range(n_rows):
        expected = unpack_binarized_boolean_array(
            binarized_data=data[i_row, :],
            n_booleans=n_cols)
        np.testing.assert_array_equal(
            expected,
            actual[i_row, :])
