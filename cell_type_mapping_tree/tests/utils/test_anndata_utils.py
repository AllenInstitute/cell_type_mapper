import pytest

import anndata
import h5py
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad,
    copy_layer_to_x)


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
        var=var)

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
        var=var)

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
        layers={'garbage': layer})

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
