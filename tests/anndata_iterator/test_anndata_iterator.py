import pytest

import anndata
import h5py
import itertools
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)


@pytest.fixture(scope='module')
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
def test_anndata_row_iterator_get_chunk(
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

    rng = np.random.default_rng(871123)
    for ii in range(5):
        i0 = rng.integers(0, 2*n_rows//3)
        i1 = i0 + rng.integers(10, 100)
        i1 = min(i1, n_rows)
        chunk = iterator.get_chunk(r0=i0, r1=i1)
        assert chunk[1] == i0
        assert chunk[2] == i1
        assert chunk[0].shape[0] == (i1-i0)
        np.testing.assert_allclose(
            chunk[0],
            x_array_fixture[i0:i1, :],
            atol=0.0,
            rtol=1.0e-7)


@pytest.mark.parametrize(
    "density,with_tmp",
    [
        ('dense', False),
        ('csr', False),
        ('csc', False),
        ('csc', True)
    ]
)
def test_anndata_iterator_from_layer(
        tmp_dir_fixture,
        density,
        with_tmp):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='anndata_from_layer_',
        suffix='.h5ad')

    n_rows = 119
    n_cols = 567
    rng = np.random.default_rng(991231)
    fraction = 0.3

    n_tot = n_rows*n_cols
    data = np.zeros(n_tot, dtype=float)
    chosen = rng.choice(
        np.arange(n_tot),
        np.round(fraction*n_tot).astype(int),
        replace=False)
    data[chosen] = rng.random(len(chosen))
    data = data.reshape((n_rows, n_cols))
    if density == 'csc':
        layer_data = scipy_sparse.csc_matrix(data)
    elif density == 'csr':
        layer_data = scipy_sparse.csr_matrix(data)
    elif density == 'dense':
        layer_data = data
    else:
        raise RuntimeError(f"cannot handle density {density}")

    a = anndata.AnnData(
        X=np.zeros((n_rows, n_cols), dtype=float),
        layers={'dummy': layer_data})

    a.write_h5ad(
        h5ad_path,
        compression='gzip',
        compression_opts=4)

    if with_tmp:
        tmp_dir = tmp_dir_fixture
    else:
        tmp_dir = None

    chunk_size = 57
    iterator = AnnDataRowIterator(
        h5ad_path=h5ad_path,
        row_chunk_size=chunk_size,
        tmp_dir=tmp_dir,
        layer='dummy')

    for r0 in range(0, n_rows, chunk_size):
        r1 = min(n_rows, r0+chunk_size)
        chunk = next(iterator)
        assert chunk[1] == r0
        assert chunk[2] == r1
        np.testing.assert_allclose(
            chunk[0],
            data[r0:r1, :],
            atol=0.0,
            rtol=1.0e-6)

    for r0, r1 in [(10, 34), (7, 67), (23, 103), (56, 119)]:
        chunk = iterator.get_chunk(r0=r0, r1=r1)
        assert chunk[1] == r0
        assert chunk[2] == r1
        np.testing.assert_allclose(
            chunk[0],
            data[r0:r1, :],
            atol=0.0,
            rtol=1.0e-6)
