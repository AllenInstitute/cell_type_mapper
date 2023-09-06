import pytest

import numpy as np
from unittest.mock import patch

from cell_type_mapper.diff_exp.scores import (
    score_differential_genes,
    penetrance_tests,
    approx_penetrance_test,
    exact_penetrance_test)



def test_penetrance_tests():

    pij_1 = np.array([0.1, 0.2, 0.3, 0.7, 0.04])
    pij_2 = np.array([0.001, 0.2, 0.002, 0.6, 0.1])

    q1_th = 0.09
    qdiff_th = 0.7
    actual = penetrance_tests(
        pij_1,
        pij_2,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        exact=True)
    expected = np.array([True, False, True, False, False])
    np.testing.assert_array_equal(actual, expected)

    q1_th = 0.09
    qdiff_th = 0.5
    actual = penetrance_tests(
        pij_1,
        pij_2,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        exact=True)
    expected = np.array([True, False, True, False, True])
    np.testing.assert_array_equal(actual, expected)

    q1_th = 0.2
    qdiff_th = 0.7
    actual = penetrance_tests(
        pij_1,
        pij_2,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        exact=True)
    expected = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(actual, expected)

    q1_th = 0.6
    qdiff_th = 0.1
    actual = penetrance_tests(
        pij_1,
        pij_2,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        exact=True)
    expected = np.array([False, False, False, True, False])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n_valid", [10, 30, 40, 70])
def test_approx_penetrance_test(n_valid):
    rng = np.random.default_rng(61232)
    q1_th = 0.5
    qdiff_th = 0.7

    n_genes = 60
    q1_score = 0.1+0.05*rng.random(n_genes)
    qdiff_score = 0.1+0.05*rng.random(n_genes)

    # designate some genes as absolutely valid
    allowable = [ii for ii in range(n_genes) if ii not in (11, 14)]
    valid = rng.choice(allowable, 23, replace=False)

    # designate some genes as absolutely invalid
    q1_score[11] = 0.01
    qdiff_score[11] = 0.8

    qdiff_score[14] = 0.01
    q1_score[14] = 0.9

    q1_score[valid] = q1_th + rng.random(len(valid))
    qdiff_score[valid] = qdiff_th + rng.random(len(valid))

    actual = approx_penetrance_test(
        q1_score=q1_score,
        qdiff_score=qdiff_score,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        n_valid=n_valid)

    # make sure absolutely invalid genes are
    # not accepted
    assert not actual[11]
    assert not actual[14]

    # make sure absolutely valid genes are accepted
    assert actual[valid].all()

    # make sure number of valid genes is as expected
    if n_valid < n_genes-2:
        n_expected = max(len(valid), n_valid)

        # slop is because of the invalid genes
        assert actual.sum() >= n_expected-2
        assert actual.sum() <= n_expected+1

    # just to be sure, make sure exact_penetrance_test does
    # what it ought to
    actual_exact = exact_penetrance_test(
        q1_score=q1_score,
        qdiff_score=qdiff_score,
        q1_th=q1_th,
        qdiff_th=qdiff_th)

    assert actual_exact[valid].all()
    assert actual_exact.sum() == len(valid)


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
            'mean': a_mean,
            'n_cells': n_a,
            'var': np.zeros(n_genes),
            'ge1': np.zeros(n_genes, dtype=int)},
        'b':{
            'mean': b_mean,
            'n_cells': n_b,
            'var': np.zeros(n_genes),
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

    module_name = 'cell_type_mapper.diff_exp.scores'
    with patch(f'{module_name}.penetrance_tests', new_penetrance):
        with patch(f'{module_name}.diffexp_p_values', new_p_values):
            (scores,
             is_marker,
             up_mask) = score_differential_genes(
                 node_1='a',
                 node_2='b',
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
