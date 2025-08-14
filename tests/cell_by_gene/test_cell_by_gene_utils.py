import pytest
import numpy as np

from cell_type_mapper.cell_by_gene.utils import (
    convert_to_cpm)


def test_convert_to_cpm():
    n_cells = 312
    n_genes = 72
    rng = np.random.default_rng(23123)
    data = rng.random((n_cells, n_genes))*2000.0
    data[11, :] = 0.0
    actual = convert_to_cpm(data)
    assert (actual[11, :] == 0.0).all
    for ii in range(n_cells):
        if ii == 11:
            continue
        row_sum = data[ii, :].sum()
        expected = data[ii, :]/row_sum
        expected = 1000000.0*expected
        np.testing.assert_allclose(
            expected,
            actual[ii, :],
            atol=0.0,
            rtol=1.0e-6)


@pytest.mark.parametrize("counts_per", [1000.0, 52.0, 7189.0])
def test_convert_to_cpm_alt_factor(counts_per):
    n_cells = 312
    n_genes = 72
    rng = np.random.default_rng(23123)
    data = rng.random((n_cells, n_genes))*2000.0
    data[11, :] = 0.0
    actual = convert_to_cpm(data, counts_per=counts_per)
    assert (actual[11, :] == 0.0).all
    for ii in range(n_cells):
        if ii == 11:
            continue
        row_sum = data[ii, :].sum()
        expected = data[ii, :]/row_sum
        expected = counts_per*expected
        np.testing.assert_allclose(
            expected,
            actual[ii, :],
            atol=0.0,
            rtol=1.0e-6)
