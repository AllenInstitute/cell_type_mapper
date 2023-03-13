import numpy as np

from hierarchical_mapping.utils.distance_utils import (
    correlation_distance,
    correlation_nearest_neighbors)



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


def test_correlation_nn():
    rng = np.random.default_rng(4455123)
    n_genes = 50
    n_baseline = 116
    n_query = 73
    baseline_array= rng.random((n_baseline, n_genes))
    query_array = rng.random((n_query, n_genes))

    actual = correlation_nearest_neighbors(
                baseline_array=baseline_array,
                query_array=query_array)

    assert actual.shape == (n_query, )
    expected = np.zeros(n_query, dtype=int)
    for i_query in range(n_query):
        mu_q = np.mean(query_array[i_query, :])
        std_q = np.std(query_array[i_query, :], ddof=0)
        min_dex = None
        min_dist = None
        for i_baseline in range(n_baseline):
            mu_b = np.mean(baseline_array[i_baseline, :])
            std_b = np.std(baseline_array[i_baseline, :], ddof=0)
            corr = np.mean((query_array[i_query, :]-mu_q)*(baseline_array[i_baseline, :]-mu_b))
            corr /= (std_q*std_b)
            dist = 1.0-corr
            if min_dex is None or dist<min_dist:
                min_dex = i_baseline
                min_dist = dist
        expected[i_query] = min_dex
    np.testing.assert_array_equal(actual, expected)
