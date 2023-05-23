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


@pytest.fixture
def n_genes():
    return 47

@pytest.fixture
def a_mean(n_genes):
    rng = np.random.default_rng(2213)
    return 5.0+rng.random(n_genes)

@pytest.fixture
def b_mean(a_mean):
    result = np.copy(a_mean)
    result[0:5] += 1.1
    result[5:12] += np.log2(1.5)+0.01
    result[20:25] -= 1.1
    result[25:35] -= np.log2(1.5)+0.01
    return result

@pytest.fixture
def n_a():
    return 15

@pytest.fixture
def n_b():
    return 27

@pytest.fixture
def summary_stats_fixture(
        n_a,
        a_mean,
        n_b,
        b_mean,
        n_genes):
    result = {
        'a':{
            'n_cells': n_a,
            'sum': n_a*a_mean,
            'sumsq': np.zeros(n_genes),
            'gt1': np.zeros(n_genes, dtype=int),
            'gt0': np.zeros(n_genes, dtype=int),
            'ge1': np.zeros(n_genes, dtype=int)},
        'b':{
            'n_cells': n_b,
            'sum': n_b*b_mean,
            'sumsq': np.zeros(n_genes),
            'gt1': np.zeros(n_genes, dtype=int),
            'gt0': np.zeros(n_genes, dtype=int),
            'ge1': np.zeros(n_genes, dtype=int)}}
    return result


@pytest.fixture
def p_values_fixture(n_genes):
    result = np.ones(n_genes, dtype=float)
    for ii in range(0, 13, 2):
        result[ii] = 0.009
    for ii in range(1, 13, 2):
        result[ii] = 0.019
    for ii in range(20, 36, 2):
        result[ii] = 0.009
    for ii in range(21, 36, 2):
        result[ii]= 0.019
    return result

@pytest.fixture
def penetrance_mask_fixture(n_genes):
    result = np.ones(n_genes, dtype=bool)
    result[1] = False
    result[2] = False
    result[10] = False
    result[11] = False
    result[21] = False
    result[22] = False
    result[27] = False
    result[28] = False
    return result


@pytest.mark.parametrize(
    "p_th, fold_change, expected_idx",
    [(0.01, 2.0, [0, 4, 20, 24]),
     (0.02, 2.0, [0, 3, 4, 20, 23, 24]),
     (0.01, 1.5, [0, 4, 6, 8, 20, 24, 26, 30, 32, 34]),
     (0.02, 1.5,
      [0, 3, 4, 5, 6, 7, 8, 9 ,20, 23, 24, 25, 26, 29,
       30, 31, 32, 33, 34])
    ])
def test_score_differential_genes_quantitatively(
        summary_stats_fixture,
        p_values_fixture,
        penetrance_mask_fixture,
        p_th,
        fold_change,
        expected_idx,
        n_genes):

    def new_penetrance(*args, **kwargs):
        return penetrance_mask_fixture
    def new_p_values(*args, **kwargs):
        return p_values_fixture

    module_name = 'hierarchical_mapping.diff_exp.scores'
    with patch(f'{module_name}.penetrance_tests', new_penetrance):
        with patch(f'{module_name}.diffexp_p_values', new_p_values):
            (scores,
             is_marker,
             up_mask) = score_differential_genes(
                 leaf_population_1=['a'],
                 leaf_population_2=['b'],
                 precomputed_stats=summary_stats_fixture,
                 p_th=p_th,
                 q1_th=0.5,
                 qdiff_th=0.7,
                 fold_change=fold_change)
    expected = np.zeros(n_genes, dtype=bool)
    expected[np.array(expected_idx)] = True
    np.testing.assert_array_equal(is_marker, expected)
    assert (up_mask[0:12] == 1).all()
    assert (up_mask[25:35] == 0).all()
