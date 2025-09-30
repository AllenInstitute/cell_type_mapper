import pytest

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist

from cell_type_mapper.utils.torch_utils import (
    is_torch_available)

from cell_type_mapper.utils.distance_utils import (
    correlation_distance,
    _correlation_nearest_neighbors_cpu,
    correlation_nearest_neighbors,
    _subtract_mean_and_normalize_cpu,
    _correlation_dot_cpu)


def test_subtract_mean_and_normalize_cpu():
    rng = np.random.default_rng(77123)
    n_cells = 14
    n_genes = 7
    data = rng.random((n_cells, n_genes))
    zeroed_out = 5
    data[zeroed_out, :] = 0.0

    actual = _subtract_mean_and_normalize_cpu(data)
    transposed_actual = _subtract_mean_and_normalize_cpu(
        data, do_transpose=True)

    for i_cell in range(n_cells):
        if i_cell == zeroed_out:
            expected = np.zeros(n_genes, dtype=float)
        else:
            mu = np.mean(data[i_cell, :])
            d = data[i_cell, :] - mu
            n = np.sqrt(np.sum(d**2))
            expected = d / n
        np.testing.assert_allclose(actual[i_cell, :], expected)
        np.testing.assert_allclose(transposed_actual[:, i_cell], expected)


def test_correlation_distance():
    rng = np.random.default_rng(87123412)
    n_genes = 50
    n0 = 116
    n1 = 73
    data0 = rng.random((n0, n_genes))
    data1 = rng.random((n1, n_genes))

    actual = correlation_distance(
                arr0=data0,
                arr1=data1)

    assert actual.shape == (n0, n1)

    scipy_expected = scipy_cdist(data0, data1, metric='correlation')
    np.testing.assert_allclose(
        actual,
        scipy_expected,
        atol=0.0,
        rtol=1.0e-6)

    expected = np.zeros((n0, n1), dtype=float)
    for i0 in range(n0):
        mu0 = np.mean(data0[i0, :])
        std0 = np.std(data0[i0, :], ddof=0)
        for i1 in range(n1):
            mu1 = np.mean(data1[i1, :])
            std1 = np.std(data1[i1, :], ddof=0)
            expected[i0, i1] = np.mean((data0[i0, :]-mu0)*(data1[i1, :]-mu1))
            expected[i0, i1] /= (std1*std0)
            expected[i0, i1] = 1.0-expected[i0, i1]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
        "return_correlation", [True, False])
def test_correlation_nn_cpu(return_correlation):
    rng = np.random.default_rng(4455123)
    n_genes = 50
    n_baseline = 116
    n_query = 73
    baseline_array = rng.random((n_baseline, n_genes))
    query_array = rng.random((n_query, n_genes))

    result = _correlation_nearest_neighbors_cpu(
                baseline_array=baseline_array,
                query_array=query_array,
                return_correlation=return_correlation)

    if return_correlation:
        actual_nn = result[0]
        actual_corr = result[1]
    else:
        actual_nn = result

    assert actual_nn.shape == (n_query, )
    expected_nn = np.zeros(n_query, dtype=int)
    expected_corr = np.zeros(n_query, dtype=float)
    for i_query in range(n_query):
        mu_q = np.mean(query_array[i_query, :])
        std_q = np.std(query_array[i_query, :], ddof=0)
        min_dex = None
        min_dist = None
        for i_baseline in range(n_baseline):
            mu_b = np.mean(baseline_array[i_baseline, :])
            std_b = np.std(baseline_array[i_baseline, :], ddof=0)
            corr = np.mean(
                (query_array[i_query, :] - mu_q)
                * (baseline_array[i_baseline, :] - mu_b)
            )
            corr /= (std_q*std_b)
            dist = 1.0-corr
            if min_dex is None or dist < min_dist:
                min_dex = i_baseline
                min_dist = dist
        expected_nn[i_query] = min_dex
        expected_corr[i_query] = 1.0-min_dist
    np.testing.assert_array_equal(actual_nn, expected_nn)
    if return_correlation:
        np.testing.assert_allclose(
            actual_corr,
            expected_corr,
            atol=0.0,
            rtol=1.0e-6)


@pytest.mark.parametrize(
        "return_correlation", [True, False])
def test_correlation_nn_runner(
        return_correlation):
    """
    Test that correlation_nearest_neighbors is consistent
    with _correlation_nearest_neigbhors_cpu
    """

    rng = np.random.default_rng(4455123)
    n_genes = 50
    n_baseline = 116
    n_query = 73
    baseline_array = rng.random((n_baseline, n_genes))
    query_array = rng.random((n_query, n_genes))

    cpu = _correlation_nearest_neighbors_cpu(
                baseline_array=baseline_array,
                query_array=query_array,
                return_correlation=return_correlation)

    test = correlation_nearest_neighbors(
                baseline_array=baseline_array,
                query_array=query_array,
                return_correlation=return_correlation)

    if return_correlation:
        np.testing.assert_array_equal(
            cpu[0], test[0])
        np.testing.assert_allclose(
            cpu[1],
            test[1],
            atol=0.0,
            rtol=1.0e-5)
    else:
        np.testing.assert_array_equal(cpu, test)
