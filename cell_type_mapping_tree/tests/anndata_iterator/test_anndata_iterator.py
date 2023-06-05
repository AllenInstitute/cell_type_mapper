import pytest

import anndata
import h5py
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('anndata_iterator'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def x_array_fixture():
    rng = np.random.default_rng(88123)
    n_rows = 1123
    n_cols = 432
    data = np.zeros(n_rows*n_cols, dtype=np.float32)
    chosen = rng.choice(
        np.arange(len(data)),
        len(data)//3,
        replace=False)
    data[chosen] = rng.random(len(chosen))
    data = data.reshape((n_rows, n_cols))
    return data


@pytest.fixture
def csr_fixture(
        tmp_dir_fixture,
        x_array_fixture):

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='csr_',
            suffix='.h5ad'))

    a = anndata.AnnData(
            X=scipy_sparse.csr_matrix(x_array_fixture))
    a.write_h5ad(h5ad_path)
    with h5py.File(h5ad_path, 'r') as src:
        attrs = dict(src['X'].attrs)
    assert attrs['encoding-type'] == 'csr_matrix'
    return h5ad_path


@pytest.fixture
def csc_fixture(
        tmp_dir_fixture,
        x_array_fixture):

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='csr_',
            suffix='.h5ad'))

    a = anndata.AnnData(
            X=scipy_sparse.csc_matrix(x_array_fixture))
    a.write_h5ad(h5ad_path)
    with h5py.File(h5ad_path, 'r') as src:
        attrs = dict(src['X'].attrs)
    assert attrs['encoding-type'] == 'csc_matrix'
    return h5ad_path


@pytest.fixture
def dense_fixture(
        tmp_dir_fixture,
        x_array_fixture):

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='csr_',
            suffix='.h5ad'))

    a = anndata.AnnData(
            X=x_array_fixture)
    a.write_h5ad(h5ad_path)

    with h5py.File(h5ad_path, 'r') as src:
        attrs = dict(src['X'].attrs)
    assert attrs['encoding-type'] == 'array'
    return h5ad_path


@pytest.mark.parametrize(
    'use, with_tmp',
    [('csr', False),
     ('csc', False),
     ('csc', True),
     ('dense', False)])
def test_anndata_row_iterator(
        x_array_fixture,
        csr_fixture,
        csc_fixture,
        dense_fixture,
        tmp_dir_fixture,
        use,
        with_tmp):
    if use == 'csr':
        fpath = csr_fixture
    elif use == 'csc':
        fpath = csc_fixture
    elif use == 'dense':
        fpath = dense_fixture
    else:
        raise RuntimeError(
            f"use={use} makese no sense")

    if with_tmp:
        tmp_dir = tmp_dir_fixture
    else:
        tmp_dir = None

    chunk_size = 123

    iterator = AnnDataRowIterator(
        h5ad_path=fpath,
        row_chunk_size=chunk_size,
        tmp_dir=tmp_dir)

    n_rows = x_array_fixture.shape[0]
    assert iterator.n_rows == n_rows
    for i0, chunk in zip(range(0, n_rows, chunk_size),
                         iterator):
        i1 = min(n_rows, i0+chunk_size)
        assert chunk[1] == i0
        assert chunk[2] == i1
        np.testing.assert_allclose(
            chunk[0],
            x_array_fixture[i0:i1, :],
            atol=0.0,
            rtol=1.0e-7)

@pytest.mark.parametrize(
    'use, with_tmp',
    [('csr', False),
     ('csc', False),
     ('csc', True),
     ('dense', False)])
def test_chunk_grab(
        x_array_fixture,
        csr_fixture,
        csc_fixture,
        dense_fixture,
        tmp_dir_fixture,
        use,
        with_tmp):
    """
    Test that the iterator lets us grab a specified chunk
    """
    if use == 'csr':
        fpath = csr_fixture
    elif use == 'csc':
        fpath = csc_fixture
    elif use == 'dense':
        fpath = dense_fixture
    else:
        raise RuntimeError(
            f"use={use} makese no sense")

    if with_tmp:
        tmp_dir = tmp_dir_fixture
    else:
        tmp_dir = None

    chunk_size = 123

    iterator = AnnDataRowIterator(
        h5ad_path=fpath,
        row_chunk_size=chunk_size,
        tmp_dir=tmp_dir)

    specified_rows = [33, 451, 2, 77, 86, 52]
    actual = iterator.get_rows(specified_rows)
    expected = x_array_fixture[specified_rows, :]
    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=1.0e-7)


@pytest.mark.parametrize(
    'use, with_tmp',
    [('csr', False),
     ('csc', False),
     ('csc', True),
     ('dense', False)])
def test_anndata_row_iterator_with_chunk_grab(
        x_array_fixture,
        csr_fixture,
        csc_fixture,
        dense_fixture,
        tmp_dir_fixture,
        use,
        with_tmp):
    """
    Test that we can also grab a chunk of specified
    rows in the middle of the iteration
    """
    if use == 'csr':
        fpath = csr_fixture
    elif use == 'csc':
        fpath = csc_fixture
    elif use == 'dense':
        fpath = dense_fixture
    else:
        raise RuntimeError(
            f"use={use} makese no sense")

    if with_tmp:
        tmp_dir = tmp_dir_fixture
    else:
        tmp_dir = None

    chunk_size = 123

    iterator = AnnDataRowIterator(
        h5ad_path=fpath,
        row_chunk_size=chunk_size,
        tmp_dir=tmp_dir)

    n_rows = x_array_fixture.shape[0]
    assert iterator.n_rows == n_rows
    ct = 0
    for i0, chunk in zip(range(0, n_rows, chunk_size),
                         iterator):
        i1 = min(n_rows, i0+chunk_size)
        assert chunk[1] == i0
        assert chunk[2] == i1
        np.testing.assert_allclose(
            chunk[0],
            x_array_fixture[i0:i1, :],
            atol=0.0,
            rtol=1.0e-7)

        if ct == 2:
            specified_rows = [10, 6, 313, 122, 11]
            actual = iterator.get_rows(specified_rows)
            np.testing.assert_allclose(
                actual,
                x_array_fixture[specified_rows, :],
                atol=0.0,
                rtol=1.0e-7)
        ct += 1
