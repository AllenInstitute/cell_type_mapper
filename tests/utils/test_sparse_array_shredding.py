"""
Test the utility that creates a single sparse array in a single
HDF5 file from a subset of rows in other sparse arrays stored
in other HDF5 files
"""
import pytest

import anndata
import h5py
import itertools
import numpy as np
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.sparse_utils import (
    amalgamate_sparse_array)


@pytest.mark.parametrize('data_dtype,verbose',
    itertools.product([np.uint8, np.uint16, np.int16, float], [True, False]))
def test_csr_amalgmation(tmp_dir_fixture, data_dtype, verbose):

    rng = np.random.default_rng(712231)
    n_cols = 15

    if data_dtype != float:
        iinfo = np.iinfo(data_dtype)
        d_max = iinfo.max
        d_min = iinfo.min
        if d_min == 0:
            d_min = 1


    src_rows = []
    expected_rows = []

    for ii in range(4):
        n_rows = rng.integers(10, 20)
        n_tot = n_rows*n_cols
        data = np.zeros(n_tot, dtype=float)
        non_null = rng.choice(
            np.arange(n_tot),
            n_tot//5,
            replace=False)

        if data_dtype == float:
            data[non_null] = rng.random(len(non_null))
        else:
            data[non_null] = rng.integers(
                d_min,
                d_max+1,
                len(non_null)).astype(float)

        data = data.reshape((n_rows, n_cols))
        chosen_rows = np.sort(rng.choice(np.arange(n_rows),
                                         rng.integers(5, 7),
                                         replace=False))
        
        # make sure some empty rows are included
        data[chosen_rows[1], :] = 0

        for idx in chosen_rows:
            expected_rows.append(data[idx, :])

        this_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad')

        a_data = anndata.AnnData(X=scipy_sparse.csr_matrix(data))
        a_data.write_h5ad(this_path)
        del a_data

        src_rows.append(
            {'path': this_path,
             'rows': list(chosen_rows)})


    expected_array = np.stack(expected_rows)

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    amalgamate_sparse_array(
        src_rows=src_rows,
        dst_path=dst_path,
        sparse_grp='X',
        verbose=verbose)

    with h5py.File(dst_path, 'r') as dst:
        actual = scipy_sparse.csr_matrix(
            (dst['X/data'][()],
             dst['X/indices'][()],
             dst['X/indptr'][()]),
            shape=expected_array.shape)

    np.testing.assert_allclose(
        actual.todense(),
        expected_array)

    assert actual.dtype == data_dtype
