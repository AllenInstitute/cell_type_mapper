import pytest

import anndata
import h5py
import itertools
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.validation.utils import (
    get_minmax_x_from_h5ad,
    round_x_to_integers,
    is_x_integers)


# add function to create various flavors of h5ad file
# with different densities, encoding types, etc.

def create_h5ad_file(
        n_rows,
        n_cols,
        max_coord,
        max_val,
        min_coord,
        min_val,
        density,
        is_chunked,
        output_path,
        int_values=False,
        int_dtype=False):

    if int_dtype:
        if not int_values:
            raise RuntimeError(
                "int_dtype is True but int_values is False")

    rng = np.random.default_rng(7612231)
    var_data = [
        {'gene_id': f'gene_{ii}', 'garbage': 'a'}
        for ii in range(n_cols)
    ]
    var = pd.DataFrame(var_data).set_index('gene_id')

    obs_data = [
        {'cell_id': f'cell_{ii}', 'bizzare': 'yes'}
        for ii in range(n_rows)
    ]
    obs = pd.DataFrame(obs_data).set_index('cell_id')

    if int_values:
        max_val = np.round(max_val)
        min_val = np.round(min_val)

    n_tot = n_rows*n_cols
    raw_data = np.zeros(n_tot, dtype=float)
    chosen = rng.choice(np.array(n_tot), n_tot//5, replace=False)
    delta = (max_val-min_val)
    if int_values:
        raw_data[chosen] = rng.integers(
                min_val+1,
                max_val-1,
                len(chosen)).astype(float)
    else:
        raw_data[chosen] = rng.random(len(chosen))*0.8*delta+min_val+0.05*delta
    raw_data = raw_data.reshape(n_rows, n_cols)
    raw_data[max_coord[0], max_coord[1]] = max_val
    raw_data[min_coord[0], min_coord[1]] = min_val

    if int_dtype:
        raw_data = raw_data.astype(int)

    assert raw_data.shape == (n_rows, n_cols)

    chunks = None
    if density == 'csr':
        data = scipy_sparse.csr_matrix(raw_data)
        if is_chunked:
            chunks = (n_tot//9, )
    elif density == 'csc':
        data = scipy_sparse.csc_matrix(raw_data)
        if is_chunked:
            chunks = (n_tot//9, )
    elif density == 'array':
        data = raw_data
        if is_chunked:
            chunks = (n_rows//3, n_cols//3)
    else:
        raise RuntimeError(
            f"do not know what density={density} means")

    a_data = anndata.AnnData(
        X=data, obs=obs, var=var, dtype=data.dtype)

    a_data.write_h5ad(output_path)

    with h5py.File(output_path, 'a') as dst:
        attrs = dict(dst['X'].attrs)
        if density == 'array':
            del dst['X']
            dataset = dst.create_dataset(
                'X',
                data=data,
                chunks=chunks)
            for k in attrs:
                dataset.attrs.create(name=k, data=attrs[k])

        else:
            del dst['X']
            g = dst.create_group('X')
            for k in attrs:
                g.attrs.create(name=k, data=attrs[k])
            g.create_dataset(
                'data',
                data=data.data,
                chunks=chunks)
            g.create_dataset(
                'indices',
                data=data.indices,
                chunks=chunks)
            g.create_dataset(
                'indptr',
                data=data.indptr)

    with h5py.File(output_path, 'r') as src:

        if density == 'array':
            data = src['X']
        else:
            data = src['X/data']

        if is_chunked:
            assert data.chunks is not None
        else:
            assert data.chunks is None

    return raw_data


@pytest.fixture(scope='session')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('validation_utils'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.mark.parametrize(
    "density,is_chunked,max_coord,min_coord",
    itertools.product(("array", "csr", "csc"),
                      (True, False),
                      ((0, 12), (54, 261), (23, 100)),
                      ((0, 7), (54, 263), (32, 122))))
def test_get_minmax(
        density,
        is_chunked,
        tmp_dir_fixture,
        max_coord,
        min_coord):

    min_val = -8.9
    max_val = 99.1

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    create_h5ad_file(
        n_rows=55,
        n_cols=267,
        max_val=max_val,
        max_coord=max_coord,
        min_val=min_val,
        min_coord=min_coord,
        density=density,
        is_chunked=is_chunked,
        output_path=output_path)

    actual = get_minmax_x_from_h5ad(output_path)
    np.testing.assert_allclose(
        actual,
        (min_val, max_val),
        atol=0.0,
        rtol=1.0e-6)


@pytest.mark.parametrize(
    "density,is_chunked",
    itertools.product(
        ('csr', 'csc', 'array'),
        (True, False)
    )
)
def test_round_x_to_integers(
        density,
        is_chunked,
        tmp_dir_fixture):

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    max_val = 101.3
    min_val = -10.2

    raw_x = create_h5ad_file(
        n_rows=55,
        n_cols=267,
        max_val=max_val,
        max_coord=(43, 112),
        min_val=min_val,
        min_coord=(21, 45),
        density=density,
        is_chunked=is_chunked,
        output_path=output_path)

    if density != 'array':
        with h5py.File(output_path, 'r') as src:
            print(dict(src['X'].attrs))
            print(src['X/data'].shape,src['X/indices'].shape,src['X/indptr'].shape)

    round_x_to_integers(
        h5ad_path=output_path,
        tmp_dir=tmp_dir_fixture)

    a_data = anndata.read_h5ad(output_path, backed='r')
    actual_x = a_data.X[()]
    if not isinstance(actual_x, np.ndarray):
        actual_x = actual_x.toarray()
    assert actual_x.dtype == int
    int_x = np.round(raw_x).astype(int)
    np.testing.assert_array_equal(actual_x, int_x)


@pytest.mark.parametrize(
    "density,is_chunked",
    itertools.product(
        ('csr', 'csc', 'array'),
        (True, False)
    )
)
def test_round_x_to_integers_no_op(
        density,
        is_chunked,
        tmp_dir_fixture):
    """
    Test case when the X axis was already an integer value
    (though dtype was a float), such that this roundint is
    a no-op
    """
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    max_val = 101.3
    min_val = -10.2

    raw_x = create_h5ad_file(
        n_rows=55,
        n_cols=267,
        max_val=max_val,
        max_coord=(43, 112),
        min_val=min_val,
        min_coord=(21, 45),
        density=density,
        is_chunked=is_chunked,
        output_path=output_path,
        int_values=True)

    if density != 'array':
        with h5py.File(output_path, 'r') as src:
            print(dict(src['X'].attrs))
            print(src['X/data'].shape,src['X/indices'].shape,src['X/indptr'].shape)

    round_x_to_integers(
        h5ad_path=output_path,
        tmp_dir=tmp_dir_fixture)

    a_data = anndata.read_h5ad(output_path, backed='r')
    actual_x = a_data.X[()]
    if not isinstance(actual_x, np.ndarray):
        actual_x = actual_x.toarray()
    assert actual_x.dtype == float
    int_x = np.round(raw_x)
    np.testing.assert_allclose(
        actual_x,
        int_x,
        atol=0.0,
        rtol=1.0e-6)



@pytest.mark.parametrize(
    "density,is_chunked,int_dtype,int_values",
    itertools.product(
        ('csr', 'csc', 'array'),
        (True, False),
        (True, False),
        (True, False)
    )
)
def test_is_x_integers(
        density,
        is_chunked,
        tmp_dir_fixture,
        int_dtype,
        int_values):
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    if int_dtype:
        if not int_values:
            return

    max_val = 101.3
    min_val = -10.2

    raw_x = create_h5ad_file(
        n_rows=55,
        n_cols=267,
        max_val=max_val,
        max_coord=(43, 112),
        min_val=min_val,
        min_coord=(21, 45),
        density=density,
        is_chunked=is_chunked,
        output_path=output_path,
        int_values=int_values,
        int_dtype=int_dtype)

    actual = is_x_integers(output_path)
    if int_values:
        assert actual
    else:
        assert not actual


@pytest.mark.parametrize(
        'is_sparse, is_int',
        itertools.product((True, False), (True, False)))
def test_is_x_integers_layers(tmp_dir_fixture, is_sparse, is_int):
    """
    Test that is_x_integers works on different
    layers in the h5ad file
    """

    rng = np.random.default_rng(223123)
    n_rows = 112
    n_cols = 73
    x = rng.random((n_rows, n_cols))
    layer = np.zeros(n_rows*n_cols, dtype=float)
    chosen = rng.choice(np.arange(n_rows*n_cols), n_rows*n_cols//3, replace=False)
    if is_int:
        layer[chosen] = rng.integers(111, 8888, len(chosen)).astype(float)
    else:
        layer[chosen] = rng.random(len(chosen))
    layer = layer.reshape((n_rows, n_cols))
    if is_sparse:
        layer = scipy_sparse.csr_matrix(layer)

    a_data = anndata.AnnData(X=x, layers={'garbage': layer}, dtype=x.dtype)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a_data.write_h5ad(h5ad_path)

    if is_int:
        assert is_x_integers(h5ad_path , layer='garbage')
    else:
        assert not is_x_integers(h5ad_path, layer='garbage')


@pytest.mark.parametrize(
        'is_sparse',
        (True, False))
def test_get_minmax_integers_layers(tmp_dir_fixture, is_sparse):
    """
    Test that is_x_integers works on different
    layers in the h5ad file
    """

    min_val = -99.9
    max_val = 89.9

    rng = np.random.default_rng(223123)
    n_rows = 112
    n_cols = 73
    x = np.zeros((n_rows, n_cols))
    layer = np.zeros(n_rows*n_cols, dtype=float)
    chosen = rng.choice(np.arange(n_rows*n_cols), n_rows*n_cols//3, replace=False)

    layer[chosen] = min_val+1.0+(max_val-1.0-min_val)*rng.random(len(chosen))

    layer[4] = min_val
    layer[5] = max_val

    layer = layer.reshape((n_rows, n_cols))
    if is_sparse:
        layer = scipy_sparse.csr_matrix(layer)

    a_data = anndata.AnnData(X=x, layers={'garbage': layer}, dtype=x.dtype)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a_data.write_h5ad(h5ad_path)

    minmax = get_minmax_x_from_h5ad(h5ad_path, layer='garbage')
    np.testing.assert_allclose(
        minmax,
        (min_val, max_val),
        atol=0.0,
        rtol=1.0e-6)
