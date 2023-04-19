import pytest

from itertools import product
import numpy as np

from hierarchical_mapping.binary_array.utils import (
    unpack_binarized_boolean_array)

from hierarchical_mapping.binary_array.binary_array import (
    BinarizedBooleanArray,
    n_int_from_n_cols)


@pytest.mark.parametrize(
        "n_rows, n_cols",
        [(9, 13), (64, 128), (57, 77), (6, 17), (17, 6), (6, 4)])
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

def test_column_setting():
    n_rows = 34
    n_cols = 71
    arr = BinarizedBooleanArray(n_rows=n_rows, n_cols=n_cols)
    rng = np.random.default_rng(55667712)
    i_col = 17
    expected = rng.integers(0, 2, n_rows).astype(bool)
    arr.set_col(i_col, expected)
    actual = arr.get_col(i_col)
    np.testing.assert_array_equal(actual, expected)

    # make sure appropriate columns get zeroed out
    arr.data = 255*np.ones(arr.data.shape, dtype=np.uint8)
    arr.set_col(i_col, expected)
    actual = arr.get_col(i_col)
    np.testing.assert_array_equal(actual, expected)

    # make sure appropriate columns get zeroed out
    arr.data = rng.integers(0, 255, (arr.n_rows, arr.n_ints), dtype=np.uint8)
    arr.set_col(i_col, expected)
    actual = arr.get_col(i_col)
    np.testing.assert_array_equal(actual, expected)


def test_column_setting_en_masse():
    rng = np.random.default_rng(55667712)
    n_rows = 34
    n_cols = 71
    src = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)
    arr = BinarizedBooleanArray(n_rows=n_rows, n_cols=n_cols)
    for i_row in range(n_rows):
        arr.set_row(i_row, data = src[i_row, :])

    been_zeroed = []
    for test_col in (17, 55, 31):
        # make sure the column wasn't already all True or all False
        assert src[:, test_col].sum() < n_cols
        assert src[:, test_col].sum() > 0
        arr.set_col_false(test_col)
        np.testing.assert_array_equal(
            arr.get_col(test_col),
            np.zeros(n_rows, dtype=bool))
        been_zeroed.append(test_col)
        for i_row in range(n_rows):
            actual = arr.get_row(i_row)
            expected = np.copy(src[i_row, :])
            expected[been_zeroed] = False
            np.testing.assert_array_equal(actual, expected)

    been_truthed = []
    for test_col in (22, 15, 3):
        # make sure the column wasn't already all True or all False
        assert src[:, test_col].sum() < n_cols
        assert src[:, test_col].sum() > 0
        arr.set_col_true(test_col)
        np.testing.assert_array_equal(
            arr.get_col(test_col),
            np.ones(n_rows, dtype=bool))
        been_truthed.append(test_col)
        for i_row in range(n_rows):
            actual = arr.get_row(i_row)
            expected = np.copy(src[i_row, :])
            expected[been_zeroed] = False
            expected[been_truthed] = True
            np.testing.assert_array_equal(actual, expected)


def test_row_setting():
    rng = np.random.default_rng(442113)
    n_cols = 34
    n_rows = 71
    src = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)
    arr = BinarizedBooleanArray(n_rows=n_rows, n_cols=n_cols)
    for i_row in range(n_rows):
        arr.set_row(i_row, data = src[i_row, :])

    been_zeroed = []
    for test_row in (17, 55, 31):
        # make sure the row wasn't already all True or all False
        assert src[test_row, :].sum() < n_rows
        assert src[test_row].sum() > 0
        arr.set_row_false(test_row)
        np.testing.assert_array_equal(
            arr.get_row(test_row),
            np.zeros(n_cols, dtype=bool))
        been_zeroed.append(test_row)
        for i_col in range(n_cols):
            actual = arr.get_col(i_col)
            expected = np.copy(src[:, i_col])
            expected[been_zeroed] = False
            np.testing.assert_array_equal(actual, expected)

    been_truthed = []
    for test_row in (22, 15, 3):
        # make sure the row wasn't already all True or all False
        assert src[test_row, :].sum() < n_rows
        assert src[test_row, :].sum() > 0
        arr.set_row_true(test_row)
        np.testing.assert_array_equal(
            arr.get_row(test_row),
            np.ones(n_cols, dtype=bool))
        been_truthed.append(test_row)
        for i_col in range(n_cols):
            actual = arr.get_col(i_col)
            expected = np.copy(src[:, i_col])
            expected[been_zeroed] = False
            expected[been_truthed] = True
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "n_cols", [17, 32, 5, 22])
def test_binary_instantiation_from_np(
        n_cols):
    n_rows = 25
    n_int = n_int_from_n_cols(n_cols)
    rng = np.random.default_rng(55123)
    src = rng.integers(0, 255, (n_rows, n_int)).astype(np.uint8)

    expected_arr = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    for i_row in range(n_rows):
        bool_arr = unpack_binarized_boolean_array(
            binarized_data=src[i_row, :],
            n_booleans=n_cols)
        expected_arr.set_row(i_row=i_row, data=bool_arr)

    actual_arr = BinarizedBooleanArray.from_data_array(
                    data_array=src,
                    n_cols=n_cols)

    for i_row in range(n_rows):
        np.testing.assert_array_equal(
            expected_arr.get_row(i_row),
            actual_arr.get_row(i_row))
    for i_col in range(n_cols):
        np.testing.assert_array_equal(
            expected_arr.get_col(i_col),
            actual_arr.get_col(i_col))


@pytest.mark.parametrize(
    "n_cols", [17, 32, 5, 22])
def test_binary_row_sums(
        n_cols):
    n_rows = 25
    rng = np.random.default_rng(55123)
    src = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)

    arr = BinarizedBooleanArray(
            n_rows=n_rows,
            n_cols=n_cols)

    for i_row in range(n_rows):
        arr.set_row(i_row, data=src[i_row, :])

    expected = src.sum(axis=0)
    actual = arr.row_sum()
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
    "n_cols", [17, 32, 5, 22])
def test_binary_col_sums(
        n_cols):
    n_rows = 25
    rng = np.random.default_rng(55123)
    src = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)

    arr = BinarizedBooleanArray(
            n_rows=n_rows,
            n_cols=n_cols)

    for i_row in range(n_rows):
        arr.set_row(i_row, data=src[i_row, :])

    expected = src.sum(axis=1)
    actual = arr.col_sum()
    np.testing.assert_array_equal(expected, actual)


def test_add_arrays():
    n_rows = 25
    n_cols = 18
    arr0 = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    arr1 = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    rng = np.random.default_rng(8812341)
    src0 = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)
    src1 = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)
    for i_row in range(n_rows):
        arr0.set_row(i_row, src0[i_row, :])
        arr1.set_row(i_row, src1[i_row, :])

    arr0.add_other(arr1)
    for i_row in range(n_rows):
        this_row = arr0.get_row(i_row)
        for i_col in range(n_cols):
            if src0[i_row, i_col] or src1[i_row, i_col]:
                assert this_row[i_col]
            else:
                assert not this_row[i_col]


@pytest.mark.parametrize(
    "col_span",
    product(range(8, 17), range(32, 41)))
def test_set_col_from_other(col_span):
    n_rows = 25
    n_cols = 57
    arr0 = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)
    arr1 = BinarizedBooleanArray(
        n_rows=n_rows,
        n_cols=n_cols)

    rng = np.random.default_rng(8812341)
    src0 = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)
    src1 = rng.integers(0, 2, (n_rows, n_cols)).astype(bool)
    for i_row in range(n_rows):
        arr0.set_row(i_row, src0[i_row, :])
        arr1.set_row(i_row, src1[i_row, :])

    col_span = (13, 31)
    arr0.copy_columns_from_other(arr1, col_span)
    for i_col in range(n_cols):
        actual = arr0.get_col(i_col)
        if i_col < col_span[0] or i_col >= col_span[1]:
            np.testing.assert_array_equal(actual, src0[:, i_col])
            assert not np.array_equal(actual, src1[:, i_col])
        else:
            np.testing.assert_array_equal(actual, src1[:, i_col])
            assert not np.array_equal(actual, src0[:, i_col])
