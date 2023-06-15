import anndata
import numpy as np
import pandas as pd
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad)


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
