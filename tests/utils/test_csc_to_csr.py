import pytest

import anndata
import h5py
import itertools
import numpy as np
import pathlib
import scipy.sparse as scipy_sparse
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.sparse_utils import (
    load_csr)

from cell_type_mapper.utils.csc_to_csr import (
    csc_to_csr_on_disk,
    re_encode_sparse_matrix_on_disk)

from cell_type_mapper.utils.csc_to_csr_parallel import (
    re_encode_sparse_matrix_on_disk_v2)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('anndata_iterator'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def x_array_fixture():
    rng = np.random.default_rng(78123)
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
def csc_fixture(
        tmp_dir_fixture,
        x_array_fixture):

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='csr_',
            suffix='.h5ad'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = anndata.AnnData(
                X=scipy_sparse.csc_matrix(x_array_fixture),
                dtype=x_array_fixture.dtype)
    a.write_h5ad(h5ad_path)
    with h5py.File(h5ad_path, 'r') as src:
        attrs = dict(src['X'].attrs)
    assert attrs['encoding-type'] == 'csc_matrix'
    return h5ad_path


@pytest.fixture
def csc_array_without_data_fixture(
        csc_fixture,
        tmp_dir_fixture):

    h5_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='no_data_',
            suffix='.h5'))

    with h5py.File(h5_path, 'w') as dst:
        with h5py.File(csc_fixture, 'r') as src:
            dst.create_dataset(
                'indices',
                data=src['X/indices'][()],
                chunks=src['X/indices'].chunks)
            dst.create_dataset(
                'indptr',
                data=src['X/indptr'][()],
                chunks=src['X/indptr'].chunks)

    return h5_path


@pytest.mark.parametrize(
    'max_gb',
    [0.1, 0.01, 0.001, 0.0001]
)
def test_csc_to_csr_on_disk(
        tmp_dir_fixture,
        x_array_fixture,
        csc_fixture,
        max_gb):

    csr_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')
    with h5py.File(csc_fixture, 'r') as original:
        csc_to_csr_on_disk(
            csc_group=original['X'],
            csr_path=csr_path,
            array_shape=x_array_fixture.shape,
            max_gb=max_gb)

    expected_csr = scipy_sparse.csr_matrix(x_array_fixture)
    with h5py.File(csr_path, 'r') as src:
        np.testing.assert_array_equal(
            src['indptr'][()], expected_csr.indptr)
        np.testing.assert_array_equal(
            src['indices'][()], expected_csr.indices)
        np.testing.assert_allclose(
            src['data'][()],
            expected_csr.data,
            atol=0.0,
            rtol=1.0e-7)

        dr = 372
        for r0 in range(0, x_array_fixture.shape[0], dr):
            r1 = min(x_array_fixture.shape[0], r0+dr)
            actual = load_csr(
                row_spec=(r0, r1),
                data=src['data'],
                indices=src['indices'],
                indptr=src['indptr'],
                n_cols=x_array_fixture.shape[1])
            expected = x_array_fixture[r0:r1, :]
            np.testing.assert_allclose(
                actual,
                expected,
                atol=0.0,
                rtol=1.0e-7)


@pytest.mark.parametrize(
        'max_gb,use_data,version',
        itertools.product(
            [0.1, 0.01, 0.001, 0.0001],
            [True, False],
            [1, 2])
)
def test_re_encode_sparse_matrix_on_disk(
        tmp_dir_fixture,
        x_array_fixture,
        csc_fixture,
        max_gb,
        use_data,
        version):

    output_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')
    if version == 1:
        with h5py.File(csc_fixture, 'r') as original:
            if use_data:
                data_handle = original['X/data']
            else:
                data_handle = None

            if version == 1:
                re_encode_sparse_matrix_on_disk(
                    indices_handle=original['X/indices'],
                    indptr_handle=original['X/indptr'],
                    data_handle=data_handle,
                    output_path=output_path,
                    indices_max=x_array_fixture.shape[0],
                    max_gb=max_gb)
    else:
        if use_data:
            data_tag = 'X/data'
        else:
            data_tag = None

        re_encode_sparse_matrix_on_disk_v2(
            h5_path=csc_fixture,
            indices_tag='X/indices',
            indptr_tag='X/indptr',
            data_tag=data_tag,
            output_path=output_path,
            indices_max=x_array_fixture.shape[0],
            max_gb=16,
            tmp_dir=tmp_dir_fixture,
            n_processors=4)

    expected_csr = scipy_sparse.csr_matrix(x_array_fixture)
    with h5py.File(output_path, 'r') as src:
        np.testing.assert_array_equal(
            src['indptr'][()], expected_csr.indptr)
        np.testing.assert_array_equal(
            src['indices'][()], expected_csr.indices)

        if use_data:
            np.testing.assert_allclose(
                src['data'][()],
                expected_csr.data,
                atol=0.0,
                rtol=1.0e-7)
        else:
            assert 'data' not in src.keys()

        if use_data:
            dr = 372
            for r0 in range(0, x_array_fixture.shape[0], dr):
                r1 = min(x_array_fixture.shape[0], r0+dr)
                actual = load_csr(
                    row_spec=(r0, r1),
                    data=src['data'],
                    indices=src['indices'],
                    indptr=src['indptr'],
                    n_cols=x_array_fixture.shape[1])
                expected = x_array_fixture[r0:r1, :]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=0.0,
                    rtol=1.0e-7)


@pytest.mark.parametrize('max_gb', [0.1, 0.01, 0.001, 0.0001])
def test_csc_to_csr_on_disk_without_data_array(
        tmp_dir_fixture,
        x_array_fixture,
        csc_array_without_data_fixture,
        max_gb):

    csr_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5')
    with h5py.File(csc_array_without_data_fixture, 'r') as original:
        csc_to_csr_on_disk(
            csc_group=original,
            csr_path=csr_path,
            array_shape=x_array_fixture.shape,
            max_gb=max_gb,
            use_data_array=False)

    expected_csr = scipy_sparse.csr_matrix(x_array_fixture)
    with h5py.File(csr_path, 'r') as src:
        np.testing.assert_array_equal(
            src['indptr'][()], expected_csr.indptr)
        np.testing.assert_array_equal(
            src['indices'][()], expected_csr.indices)


@pytest.mark.parametrize(
    "indices_slice",
    [
        (0, 155),
        (10, 56),
        (2, 100),
        (56, 155),
        (0, 97)
    ]
)
def test_re_encode_subset_of_matrix(
        indices_slice,
        tmp_dir_fixture):
    """
    Test transposing sparse matrix on disk, but only keeping a
    slice of indices contents
    """
    rng = np.random.default_rng(77123)
    nrows = 155
    ncols = 245
    ntot = nrows*ncols
    data = np.zeros(ntot, dtype=np.uint8)
    chosen = rng.choice(np.arange(ntot), ntot//5, replace=False)
    data[chosen] = rng.integers(1, 255, len(chosen))
    data = data.reshape(nrows, ncols)
    data_csc = scipy_sparse.csc_array(data)

    csc_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csc_',
        suffix='.h5')

    with h5py.File(csc_path, 'w') as dst:
        dst.create_dataset(
            'data',
            data=data_csc.data,
            chunks=(1000,))
        dst.create_dataset(
            'indices',
            data=data_csc.indices,
            chunks=(1000,))
        dst.create_dataset(
            'indptr',
            data=data_csc.indptr)

    csr_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csr_',
        suffix='.h5')

    with h5py.File(csc_path, 'r') as src:
        re_encode_sparse_matrix_on_disk(
            indices_handle=src['indices'],
            indptr_handle=src['indptr'],
            data_handle=src['data'],
            indices_max=nrows,
            max_gb=0.001,
            output_path=csr_path,
            indices_slice=indices_slice)

    with h5py.File(csr_path, 'r') as src:
        actual = scipy_sparse.csr_array(
            (src['data'][()], src['indices'][()], src['indptr'][()]),
            shape=(indices_slice[1]-indices_slice[0], ncols))

    expected = scipy_sparse.csr_array(
        data[indices_slice[0]:indices_slice[1], :])

    np.testing.assert_array_equal(
        actual.indptr,
        expected.indptr)

    np.testing.assert_array_equal(
        actual.indices,
        expected.indices)

    np.testing.assert_array_equal(
        actual.data,
        expected.data)

    actual = actual.toarray()
    np.testing.assert_array_equal(
        actual,
        data[indices_slice[0]:indices_slice[1], :])
