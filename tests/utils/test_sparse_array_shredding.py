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
import pandas as pd
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.anndata_utils import (
    amalgamate_h5ad)


@pytest.mark.parametrize('data_dtype, layer, density, dst_sparse',
    itertools.product(
        [np.uint8, np.uint16, np.int16, float],
        ['X', 'dummy', 'raw/X'],
        ['csr', 'csc', 'dense'],
        [True, False]))
def test_csr_amalgamation(
        tmp_dir_fixture,
        data_dtype,
        layer,
        density,
        dst_sparse):

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

        if density == 'csr':
            data = scipy_sparse.csr_matrix(data.astype(data_dtype))
        elif density == 'csc':
            data = scipy_sparse.csc_matrix(data.astype(data_dtype))
        else:
            data = data.astype(data_dtype)

        if layer == 'X':
            x = data
            layers = None
            raw = None
        elif layer == 'dummy':
            x = np.zeros(data.shape, dtype=int)
            layers = {layer: data}
            raw = None
        elif layer == 'raw/X':
            x = np.zeros(data.shape, dtype=int)
            layers = None
            raw = {'X': data}
        else:
            raise RuntimeError(
                f"Test does not know how to parse layer '{layer}'"
            )

        a_data = anndata.AnnData(
            X=x,
            layers=layers,
            raw=raw,
            dtype=data_dtype)

        a_data.write_h5ad(this_path)

        del a_data

        src_rows.append(
            {'path': this_path,
             'rows': list(chosen_rows),
             'layer': layer})

    expected_array = np.stack(expected_rows)

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    dst_var = pd.DataFrame(
        [{'g': f'g_{ii}'}
         for ii in range(n_cols)]).set_index('g')
    dst_obs = pd.DataFrame(
        [{'c': f'c_{ii}'}
         for ii in range(expected_array.shape[0])]).set_index('c')

    amalgamate_h5ad(
        src_rows=src_rows,
        dst_path=dst_path,
        dst_var=dst_var,
        dst_obs=dst_obs,
        dst_sparse=dst_sparse,
        tmp_dir=tmp_dir_fixture)

    if dst_sparse:

        with h5py.File(dst_path, 'r') as dst:
            assert dst['X/indices'].dtype == np.int32
            assert dst['X/indptr'].dtype == np.int32
            actual = scipy_sparse.csr_matrix(
                (dst['X/data'][()],
                 dst['X/indices'][()],
                 dst['X/indptr'][()]),
                shape=expected_array.shape)
            actual = actual.toarray()

    else:
        with h5py.File(dst_path, 'r') as dst:
            actual = dst['X'][()]

    np.testing.assert_allclose(
        actual,
        expected_array)

    assert actual.dtype == data_dtype


@pytest.mark.parametrize('data_dtype, layer, density, dst_sparse',
    itertools.product(
        [np.uint8, np.uint16, np.int16, float],
        ['X', 'dummy', 'raw/X'],
        ['csr', 'csc', 'dense'],
        [True, False]))
def test_csr_anndata_amalgamation(
        tmp_dir_fixture,
        data_dtype,
        layer,
        density,
        dst_sparse):

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

        if density == 'csr':
            data = scipy_sparse.csr_matrix(data.astype(data_dtype))
        elif density == 'csc':
            data = scipy_sparse.csc_matrix(data.astype(data_dtype))
        else:
            data = data.astype(data_dtype)

        if layer == 'X':
            x = data
            layers = None
            raw = None
        elif layer == 'dummy':
            x = np.zeros(data.shape, dtype=int)
            layers = {layer: data}
            raw = None
        elif layer == 'raw/X':
            x = np.zeros(data.shape, dtype=int)
            layers = None
            raw = {'X': data}
        else:
            raise RuntimeError(
                f"Test does not know how to parse layer '{layer}'"
            )

        a_data = anndata.AnnData(
            X=x,
            layers=layers,
            raw=raw,
            dtype=data_dtype)

        a_data.write_h5ad(this_path)

        del a_data

        src_rows.append(
            {'path': this_path,
             'rows': list(chosen_rows),
             'layer': layer})

    expected_array = np.stack(expected_rows)

    new_obs = pd.DataFrame(
        [{'cell_id': f'c_{ii}', 'junk': ii**2}
         for ii in range(expected_array.shape[0])]).set_index('cell_id')

    new_var = pd.DataFrame(
        [{'gene': f'g_{ii}', 'garbage': ii**3}
         for ii in range(expected_array.shape[1])]).set_index('gene')

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    amalgamate_h5ad(
        src_rows=src_rows,
        dst_path=dst_path,
        dst_obs=new_obs,
        dst_var=new_var,
        dst_sparse=dst_sparse)

    actual_a = anndata.read_h5ad(dst_path, backed='r')
    pd.testing.assert_frame_equal(actual_a.obs, new_obs)
    pd.testing.assert_frame_equal(actual_a.var, new_var)

    if dst_sparse:
        actual_x = actual_a.X[()].todense()
    else:
        actual_x = actual_a.X[()]

    np.testing.assert_allclose(
        actual_x,
        expected_array)

    assert actual_x.dtype == data_dtype

    # test that the resulting anndata object can be sliced on columns
    col_idx = [2, 8, 1, 11]
    col_pd_idx = actual_a.var.index[col_idx]
    actual = actual_a[:, col_pd_idx].to_memory()
    expected = expected_array[:, col_idx]
    np.testing.assert_allclose(
        actual.chunk_X(np.arange(len(actual_a.obs))),
        expected,
        atol=0.0,
        rtol=1.0e-6)


@pytest.mark.parametrize('layer', ['X', 'dummy', 'raw/X'])
def test_failure_when_many_floats(tmp_dir_fixture, layer):
    """
    Test that amalgamation fails when the input arrays
    have disparate float dtypes
    """
    rng = np.random.default_rng(712231)
    n_cols = 15


    src_rows = []
    expected_rows = []

    for ii, data_dtype in enumerate(
                [np.float32, np.float64, np.float32]):
        n_rows = rng.integers(10, 20)
        n_tot = n_rows*n_cols
        data = np.zeros(n_tot, dtype=float)
        non_null = rng.choice(
            np.arange(n_tot),
            n_tot//5,
            replace=False)

        data[non_null] = rng.random(len(non_null), dtype=data_dtype)
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

        if layer == 'X':
            x = scipy_sparse.csr_matrix(data)
            layers = None
            raw = None
        elif layer == 'dummy':
            x = np.zeros(data.shape, dtype=int)
            layers = {layer: scipy_sparse.csr_matrix(data.astype(data_dtype))}
            raw = None
        elif layer == 'raw/X':
            x = np.zeros(data.shape, dtype=int)
            layers = None
            raw = {'X': scipy_sparse.csr_matrix(data.astype(data_dtype))}
        else:
            raise RuntimeError(
                f"Test does not know how to parse layer '{layer}'"
            )

        a_data = anndata.AnnData(
            X=x,
            layers=layers,
            raw=raw,
            dtype=data_dtype)

        a_data.write_h5ad(this_path)

        del a_data

        src_rows.append(
            {'path': this_path,
             'rows': list(chosen_rows),
             'layer': layer})

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    with pytest.raises(RuntimeError, match="disparate data types"):
        amalgamate_h5ad(
            src_rows=src_rows,
            dst_path=dst_path,
            dst_obs=None,
            dst_var=None)
