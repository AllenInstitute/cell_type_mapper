import numpy as np
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.stats_utils import (
    summary_stats_for_chunk)


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
    csc = scipy_sparse.csc_array(data)

    actual = summary_stats_for_chunk(
                cell_x_gene=csc)

    assert actual['n_cells'] == nrows
    assert actual['sum'].shape == (ncols,)
    np.testing.assert_array_equal(actual['sum'], data.sum(axis=0))
    assert actual['sumsq'].shape == (ncols,)
    np.testing.assert_array_equal(actual['sumsq'], (data**2).sum(axis=0))
    assert actual['gt0'].shape == (ncols,)
    np.testing.assert_array_equal(actual['gt0'], (data>0).sum(axis=0))
    assert actual['gt1'].shape == (ncols,)
    np.testing.assert_array_equal(actual['gt1'], (data>1).sum(axis=0))
