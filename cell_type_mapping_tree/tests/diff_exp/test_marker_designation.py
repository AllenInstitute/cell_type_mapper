import pytest

import numpy as np
from unittest.mock import patch

from hierarchical_mapping.diff_exp.scores import (
    score_differential_genes,
    penetrance_tests)



def test_penetrance_tests():

    pij_1 = np.array([0.1, 0.2, 0.3, 0.7, 0.04])
    pij_2 = np.array([0.001, 0.2, 0.002, 0.6, 0.1])

    q1_th = 0.09
    qdiff_th = 0.7
    actual = penetrance_tests(pij_1, pij_2, q1_th=q1_th, qdiff_th=qdiff_th)
    expected = np.array([True, False, True, False, False])
    np.testing.assert_array_equal(actual, expected)

    q1_th = 0.09
    qdiff_th = 0.5
    actual = penetrance_tests(pij_1, pij_2, q1_th=q1_th, qdiff_th=qdiff_th)
    expected = np.array([True, False, True, False, True])
    np.testing.assert_array_equal(actual, expected)

    q1_th = 0.2
    qdiff_th = 0.7
    actual = penetrance_tests(pij_1, pij_2, q1_th=q1_th, qdiff_th=qdiff_th)
    expected = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(actual, expected)

    q1_th = 0.6
    qdiff_th = 0.1
    actual = penetrance_tests(pij_1, pij_2, q1_th=q1_th, qdiff_th=qdiff_th)
    expected = np.array([False, False, False, True, False])
    np.testing.assert_array_equal(actual, expected)
