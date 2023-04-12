import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist

from hierarchical_mapping.utils.distance_utils import (
    correlation_distance,
    correlation_nearest_neighbors,
    _subtract_mean_and_normalize,
    cosine_distance,
    cosine_nearest_neighbors)


def test_subtract_mean_and_normalize():
    rng = np.random.default_rng(77123)
    n_cells = 14
    n_genes = 7
    data = rng.random((n_cells, n_genes))
    zeroed_out = 5
    data[zeroed_out, :] = 0.0

    actual = _subtract_mean_and_normalize(data)
    transposed_actual = _subtract_mean_and_normalize(data, do_transpose=True)

    for i_cell in range(n_cells):
        if i_cell == zeroed_out:
            expected = np.zeros(n_genes, dtype=float)
        else:
            mu = np.mean(data[i_cell, :])
            d = data[i_cell, :] - mu
            n = np.sqrt(np.sum(d**2))
            expected = d /n
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


def test_cosine_distance():
    rng = np.random.default_rng(221314)
    ref = rng.random((112, 34))
    query = rng.random((17, 34))

    expected = scipy_cdist(
            query,
            ref,
            metric='cosine')

    actual = cosine_distance(
            arr0=query,
            arr1=ref)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=1.0e-6)


def test_cosine_nearest_neighbors():
    rng = np.random.default_rng(221314)
    ref = rng.random((112, 34))
    query = rng.random((17, 34))

    actual = cosine_nearest_neighbors(
            baseline_array=ref,
            query_array=query)

    true_dist = scipy_cdist(query, ref, metric='cosine')
    assert true_dist.shape == (query.shape[0], ref.shape[0])
    assert actual.shape == (query.shape[0], )
    for i_row in range(query.shape[0]):
        assert actual[i_row] == np.argmin(true_dist[i_row, :])
