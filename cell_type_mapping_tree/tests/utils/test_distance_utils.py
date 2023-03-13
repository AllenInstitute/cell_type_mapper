import numpy as np

from hierarchical_mapping.utils.distance_utils import (
    correlation_distance)



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

