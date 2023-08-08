import pytest

import anndata
import h5py
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad,
    copy_layer_to_x,
    read_uns_from_h5ad,
    write_uns_to_h5ad,
    append_to_obsm)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('anndata_utils_'))
    yield tmp_dir
    _clean_up(tmp_dir)


def test_read_df(
        tmp_path_factory):
    obs_data = [
        {'id': 'a', 'junk': 2},
        {'id': 'b', 'junk': 9},
        {'id': 'c', 'junk': 12}]
    obs = pd.DataFrame(obs_data)
    obs = obs.set_index('id')

    var_data = [
        {'var_id': 'aa', 'silly': 22},
        {'var_id': 'bb', 'silly': 92},
        {'var_id': 'cc', 'silly': 122},
        {'var_id': 'dd', 'silly': 99}]
    var = pd.DataFrame(var_data)
    var = var.set_index('var_id')

    ad = anndata.AnnData(
        X=np.random.random((3,4)),
        obs=obs,
        var=var,
        dtype=float)

    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('anndata_reader'))
    tmp_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.h5ad')

    ad.write_h5ad(tmp_path)

    actual_obs = read_df_from_h5ad(tmp_path, 'obs')
    pd.testing.assert_frame_equal(obs, actual_obs)

    actual_var = read_df_from_h5ad(tmp_path, 'var')
    pd.testing.assert_frame_equal(var, actual_var)
    _clean_up(tmp_dir)


def test_write_df(
        tmp_path_factory):
    obs_data = [
        {'id': 'a', 'junk': 2},
        {'id': 'b', 'junk': 9},
        {'id': 'c', 'junk': 12}]
    obs = pd.DataFrame(obs_data)
    obs = obs.set_index('id')

    var_data = [
        {'var_id': 'aa', 'silly': 22},
        {'var_id': 'bb', 'silly': 92},
        {'var_id': 'cc', 'silly': 122},
        {'var_id': 'dd', 'silly': 99}]
    var = pd.DataFrame(var_data)
    var = var.set_index('var_id')

    x = np.random.random((3,4))

    ad = anndata.AnnData(
        X=np.random.random((3,4)),
        obs=obs,
        var=var,
        dtype=float)

    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('anndata_reader'))
    tmp_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.h5ad')

    ad.write_h5ad(tmp_path)

    other_var_data = [
        {'var_id': 'asa', 'silly': 22, 'garbage': '45u'},
        {'var_id': 'btb', 'silly': 92, 'garbage': '71y'},
        {'var_id': 'cec', 'silly': 122, 'garbage': '8c'},
        {'var_id': 'd3d', 'silly': 99, 'garbage': '7x'}]

    other_var = pd.DataFrame(other_var_data).set_index('garbage')

    write_df_to_h5ad(
        h5ad_path=tmp_path,
        df_name='var',
        df_value=other_var)

    actual = anndata.read_h5ad(tmp_path, backed='r')
    pd.testing.assert_frame_equal(obs, actual.obs)
    np.testing.assert_allclose(
        actual.X[()],
        x,
        atol=0.0,
        rtol=1.0e6)
    pd.testing.assert_frame_equal(
        actual.var,
        other_var)


@pytest.mark.parametrize("is_sparse", [True, False])
def test_copy_layer_to_x(is_sparse, tmp_dir_fixture):

    rng = np.random.default_rng(2231)
    n_rows = 45
    n_cols = 17
    x = np.zeros((n_rows, n_cols))
    obs = pd.DataFrame(
        [{'a': str(ii), 'b': ii**2} for ii in range(n_rows)]).set_index('a')
    var = pd.DataFrame(
        [{'c': str(ii**3), 'd': ii*0.8} for ii in range(n_cols)]).set_index('c')
    layer = np.zeros(n_rows*n_cols, dtype=float)
    chosen = rng.choice(np.arange(n_rows*n_cols),
                        n_rows*n_cols//3,
                        replace=False)

    layer[chosen] = rng.random(len(chosen))
    layer = layer.reshape((n_rows, n_cols))
    expected = np.copy(layer)
    if is_sparse:
        layer = scipy_sparse.csr_matrix(layer)

    a_data = anndata.AnnData(
        X=x,
        obs=obs,
        var=var,
        layers={'garbage': layer},
        dtype=x.dtype)

    baseline_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a_data.write_h5ad(baseline_path)

    other_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    copy_layer_to_x(
        original_h5ad_path=baseline_path,
        new_h5ad_path=other_path,
        layer='garbage')

    actual = anndata.read_h5ad(other_path, backed='r')
    pd.testing.assert_frame_equal(obs, actual.obs)
    pd.testing.assert_frame_equal(var, actual.var)
    if is_sparse:
        actual_x = actual.X[()].toarray()
    else:
        actual_x = actual.X[()]

    np.testing.assert_allclose(
        actual_x,
        expected,
        atol=0.0,
        rtol=1.0e-6)

    with h5py.File(baseline_path, 'r') as baseline:
        with h5py.File(other_path, 'r') as test:
            baseline_attrs = dict(baseline['layers/garbage'].attrs)
            test_attrs = dict(test['X'].attrs)
            assert set(baseline_attrs.keys()) == set(test_attrs.keys())
            for k in baseline_attrs:
                b = baseline_attrs[k]
                t = test_attrs[k]
                if not isinstance(b, np.ndarray):
                    assert b == t
                else:
                    assert (b==t).all()


def test_read_write_uns_from_h5ad(tmp_dir_fixture):
    """
    Test utility to read unstructured metadata from
    h5ad file.

    And utility to write unstructured metadata to h5ad file
    """
    uns = {'a': 1, 'b': 2}
    a_data = anndata.AnnData(
        X=np.random.random_sample((12, 27)),
        uns=uns,
        dtype=float)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a_data.write_h5ad(h5ad_path)

    actual = read_uns_from_h5ad(h5ad_path)
    assert actual == uns

    uns['c'] = 'abcdefg'
    write_uns_to_h5ad(h5ad_path, uns)

    b_data = anndata.read_h5ad(h5ad_path, backed='r')
    assert b_data.uns == uns


def test_read_empty_uns(tmp_dir_fixture):
    """
    Make sure that reading uns from an h5ad file that
    does not have one results in an empty dict (instead
    of, say, None)
    """
    a_data = anndata.AnnData(
        X=np.zeros((5,4)),
        obs=pd.DataFrame([{'a':ii} for ii in range(5)]),
        var=pd.DataFrame([{'b':ii} for ii in range(4)]),
        dtype=float)
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)
    actual = read_uns_from_h5ad(h5ad_path)
    assert isinstance(actual, dict)
    assert len(actual) == 0


def test_append_to_obsm(tmp_dir_fixture):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='adata_with_obsm_',
        suffix='.h5ad')

    rng = np.random.default_rng(6123412)

    n_obs = 25
    n_var = 32
    expected_obsm = {
        'a': rng.integers(9, 122, (n_obs, 3)),
        'b': rng.integers(9, 122, (n_obs, 2))
    }
    a_data = anndata.AnnData(
        X=rng.random((n_obs, n_var)),
        obsm=expected_obsm)

    a_data.write_h5ad(h5ad_path)

    # make sure cannot overwrite with clobber=False
    with pytest.raises(RuntimeError, match='already in obsm'):
        append_to_obsm(
            h5ad_path=h5ad_path,
            obsm_key='a',
            obsm_value=np.zeros((n_obs, 4), dtype=int),
            clobber=False)

    expected_c = rng.integers(14, 38, (n_obs, 6))
    append_to_obsm(
        h5ad_path=h5ad_path,
        obsm_key='c',
        obsm_value=expected_c,
        clobber=False)

    roundtrip = anndata.read_h5ad(h5ad_path)
    actual_obsm = roundtrip.obsm
    np.testing.assert_array_equal(
        actual_obsm['a'],
        expected_obsm['a'])
    np.testing.assert_array_equal(
        actual_obsm['b'],
        expected_obsm['b'])
    np.testing.assert_array_equal(
        actual_obsm['c'],
        expected_c)

    # use clobber
    new_a = rng.integers(12, 99, (n_obs, 13))
    append_to_obsm(
        h5ad_path=h5ad_path,
        obsm_key='a',
        obsm_value=new_a,
        clobber=True)

    roundtrip = anndata.read_h5ad(h5ad_path)
    actual_obsm = roundtrip.obsm

    np.testing.assert_array_equal(
        actual_obsm['a'],
        new_a)
    np.testing.assert_array_equal(
        actual_obsm['b'],
        expected_obsm['b'])
    np.testing.assert_array_equal(
        actual_obsm['c'],
        expected_c)
