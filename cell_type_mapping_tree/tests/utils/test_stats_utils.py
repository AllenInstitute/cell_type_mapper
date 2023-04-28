import numpy as np

from hierarchical_mapping.utils.stats_utils import (
    summary_stats_for_chunk,
    welch_t_test)

from hierarchical_mapping.cell_by_gene.utils import (
    convert_to_cpm)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


def test_summary_stats_for_chunk():
    rng = np.random.default_rng(11235813)
    nrows = 100
    ncols = 516
    raw_data = np.zeros(nrows*ncols, dtype=int)
    chosen_dex = rng.choice(np.arange(nrows*ncols),
                            (nrows*ncols)//13,
                            replace=False)
    raw_data[chosen_dex] = rng.integers(1, 2000, len(chosen_dex))
    raw_data = raw_data.reshape(nrows, ncols)

    cpm_data = convert_to_cpm(raw_data)
    log2cpm_data = np.log(cpm_data+1.0)/np.log(2.0)

    cell_x_gene = CellByGeneMatrix(
            data=raw_data,
            gene_identifiers=[f"{ii}" for ii in range(raw_data.shape[1])],
            normalization="raw")

    cell_x_gene.to_log2CPM_in_place()

    actual = summary_stats_for_chunk(
                cell_x_gene=cell_x_gene)

    assert actual['n_cells'] == nrows

    assert actual['sum'].shape == (ncols,)
    np.testing.assert_allclose(
        actual['sum'],
        log2cpm_data.sum(axis=0),
        atol=0.0,
        rtol=1.0e-6)

    assert actual['sumsq'].shape == (ncols,)
    np.testing.assert_allclose(
        actual['sumsq'],
        (log2cpm_data**2).sum(axis=0),
        atol=0.0,
        rtol=1.0e-6)

    assert actual['gt0'].shape == (ncols,)
    assert actual['gt0'].sum() > 0
    assert actual['gt0'].sum() < nrows*ncols
    np.testing.assert_array_equal(actual['gt0'], (cpm_data>0).sum(axis=0))

    assert actual['gt1'].shape == (ncols,)
    assert actual['gt1'].sum() > 0
    assert actual['gt1'].sum() < nrows*ncols
    np.testing.assert_array_equal(actual['gt1'], (cpm_data>1).sum(axis=0))


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
