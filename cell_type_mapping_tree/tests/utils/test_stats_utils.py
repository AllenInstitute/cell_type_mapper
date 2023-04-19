import numpy as np

from hierarchical_mapping.utils.stats_utils import (
    summary_stats_for_chunk,
    welch_t_test)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


def test_summary_stats_for_chunk():
    rng = np.random.default_rng(11235813)
    nrows = 100
    ncols = 516
    data = np.zeros(nrows*ncols, dtype=int)
    chosen_dex = rng.choice(np.arange(nrows*ncols),
                            (nrows*ncols)//13,
                            replace=False)
    data[chosen_dex] = rng.integers(1, 2000, len(chosen_dex))
    data = data.reshape(nrows, ncols)

    cell_x_gene = CellByGeneMatrix(
            data=data,
            gene_identifiers=[f"{ii}" for ii in range(data.shape[1])],
            normalization="log2CPM")

    actual = summary_stats_for_chunk(
                cell_x_gene=cell_x_gene)

    assert actual['n_cells'] == nrows
    assert actual['sum'].shape == (ncols,)
    np.testing.assert_array_equal(actual['sum'], data.sum(axis=0))
    assert actual['sumsq'].shape == (ncols,)
    np.testing.assert_array_equal(actual['sumsq'], (data**2).sum(axis=0))
    assert actual['gt0'].shape == (ncols,)
    np.testing.assert_array_equal(actual['gt0'], (data>0).sum(axis=0))
    assert actual['gt1'].shape == (ncols,)
    np.testing.assert_array_equal(actual['gt1'], (data>1).sum(axis=0))


def test_welch_t_test():
    """
    Just tests that calling it on numpy arrays works
    on the arguments element-wise the way you would expect
    """
    n_cols = 12
    rng = np.random.default_rng(66123)

    mean1 = rng.random(n_cols)
    var1 = rng.random(n_cols)
    n1 = rng.integers(25, 75)
    mean2 = rng.random(n_cols)
    var2 = rng.random(n_cols)
    n2 = rng.integers(25, 75)

    actual = welch_t_test(
                mean1=mean1,
                var1=var1,
                n1=n1,
                mean2=mean2,
                var2=var2,
                n2=n2)

    for idx in range(n_cols):
        expected = welch_t_test(
                        mean1=mean1[idx],
                        var1=var1[idx],
                        n1=n1,
                        mean2=mean2[idx],
                        var2=var2[idx],
                        n2=n2)

        np.testing.assert_allclose(expected[0], actual[0][idx])
        np.testing.assert_allclose(expected[1], actual[1][idx])
        np.testing.assert_allclose(expected[2], actual[2][idx])
