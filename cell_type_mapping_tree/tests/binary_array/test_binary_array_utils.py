import pytest

import numpy as np

from hierarchical_mapping.binary_array.utils import (
    binarize_boolean_array,
    unpack_binarized_boolean_array)



@pytest.mark.parametrize(
        "n_bool", [8, 32, 64, 57, 13, 23])
def test_roundtrip(n_bool):
    rng = np.random.default_rng(6677122)
    src = rng.integers(0, 2, n_bool).astype(bool)
    binarized = binarize_boolean_array(src)
    actual = unpack_binarized_boolean_array(
        binarized_data=binarized,
        n_booleans=n_bool)
    np.testing.assert_array_equal(src, actual)
    assert binarized.dtype == np.uint8
    assert len(binarized) == np.ceil(n_bool/8).astype(int)
