import pytest

import anndata
import hashlib
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.cli.validate_h5ad import (
    ValidateH5adRunner)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('validating_h5ad_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def var_fixture():
    records = [
        {'gene_id': 'Clec10a', 'val': 'ab'},
        {'gene_id': 'Alox12', 'val': 'cd'},
        {'gene_id': 'Comt', 'val': 'xy'},
        {'gene_id': 'hammer', 'val': 'uw'}]

    return pd.DataFrame(records).set_index('gene_id')

@pytest.fixture
def good_var_fixture():
    records = [
        {'gene_id': "ENSMUSG00000000308", 'val': 'ab'},
        {'gene_id': "ENSMUSG00000000326", 'val': 'cd'},
        {'gene_id': "ENSMUSG00000000276", 'val': 'xy'},
        {'gene_id': "ENSMUSG00000000248", 'val': 'uw'}]

    return pd.DataFrame(records).set_index('gene_id')


@pytest.fixture
def obs_fixture():
    records = [
        {'cell_id': f'cell_{ii}', 'sq': ii**2}
        for ii in range(5)]
    return pd.DataFrame(records).set_index('cell_id')


@pytest.fixture
def x_fixture(var_fixture, obs_fixture):
    n_rows = len(obs_fixture)
    n_cols = len(var_fixture)
    n_tot = n_rows*n_cols
    data = np.zeros((n_rows, n_cols), dtype=float)
    rng = np.random.default_rng(77123)
    for i_row in range(n_rows):
        chosen = np.unique(rng.integers(0, n_cols, 2))
        for i_col in chosen:
            data[i_row, i_col] = rng.random()*10.0+1.4
    return data

@pytest.fixture
def good_x_fixture(var_fixture, obs_fixture):
    n_rows = len(obs_fixture)
    n_cols = len(var_fixture)
    n_tot = n_rows*n_cols
    data = np.zeros((n_rows, n_cols), dtype=float)
    rng = np.random.default_rng(77123)
    for i_row in range(n_rows):
        chosen = np.unique(rng.integers(0, n_cols, 2))
        for i_col in chosen:
            data[i_row, i_col] = np.round(rng.random()*10.0+1.4)
    return data


@pytest.mark.parametrize(
        "density", ("csr", "csc", "array"))
def test_validation_cli_of_h5ad(
        var_fixture,
        obs_fixture,
        x_fixture,
        tmp_dir_fixture,
        density):

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    if density == "array":
        x = x_fixture
    elif density == "csr":
        x = scipy_sparse.csr_matrix(x_fixture)
    elif density == "csc":
        x = scipy_sparse.csc_matrix(x_fixture)
    else:
        raise RuntimeError(f"unknown density {density}")

    a_data = anndata.AnnData(X=x, var=var_fixture, obs=obs_fixture)
    a_data.write_h5ad(orig_path)

    md50 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md50.update(src.read())

    output_json = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix=f"bad_input_{density}_",
        suffix=".json")

    config = {
        'h5ad_path': orig_path,
        'output_dir': str(tmp_dir_fixture.resolve().absolute()),
        'tmp_dir': str(tmp_dir_fixture.resolve().absolute()),
        'output_json': output_json
    }
    
    runner = ValidateH5adRunner(args=[], input_data=config)
    runner.run()

    output_manifest = json.load(open(output_json, 'rb'))
    result_path = output_manifest['valid_h5ad_path']

    with h5py.File(result_path, 'r') as in_file:
        if density != 'array':
            data_key = 'X/data'
        else:
            data_key = 'X'
        assert in_file[data_key].dtype == np.uint8

    actual = anndata.read_h5ad(result_path, backed='r')
    pd.testing.assert_frame_equal(obs_fixture, actual.obs)
    actual_x = actual.X[()]
    if density != "array":
        actual_x = actual_x.toarray()
    assert not np.allclose(actual_x, x_fixture)
    assert np.array_equal(
        actual_x,
        np.round(x_fixture).astype(int))

    actual_var = actual.var
    assert len(actual_var.columns) == 2
    assert list(actual_var['gene_id'].values) == list(var_fixture.index.values)
    assert list(actual_var['val'].values) == list(var_fixture.val.values)
    actual_idx = list(actual_var.index.values)
    assert len(actual_idx) == 4
    assert actual_idx[0] == "ENSMUSG00000000318"
    assert actual_idx[1] == "ENSMUSG00000000320"
    assert actual_idx[2] == "ENSMUSG00000000326"
    assert "nonsense" in actual_idx[3]

    # make sure input file did not change
    md51 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md51.update(src.read())

    assert md50.hexdigest() == md51.hexdigest()

    for k in config:
        assert output_manifest['config'][k] == config[k]

    assert len(output_manifest['log_messages']) > 0


@pytest.mark.parametrize(
        "density", ("csr", "csc", "array"))
def test_validation_cli_of_good_h5ad(
        good_var_fixture,
        obs_fixture,
        good_x_fixture,
        tmp_dir_fixture,
        density):

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    if density == "array":
        x = good_x_fixture
    elif density == "csr":
        x = scipy_sparse.csr_matrix(good_x_fixture)
    elif density == "csc":
        x = scipy_sparse.csc_matrix(good_x_fixture)
    else:
        raise RuntimeError(f"unknown density {density}")

    a_data = anndata.AnnData(X=x, var=good_var_fixture, obs=obs_fixture)
    a_data.write_h5ad(orig_path)

    md50 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md50.update(src.read())

    output_json = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix=f"good_input_{density}_",
        suffix=".json")

    config = {
        'h5ad_path': orig_path,
        'output_dir': str(tmp_dir_fixture.resolve().absolute()),
        'tmp_dir': str(tmp_dir_fixture.resolve().absolute()),
        'output_json': output_json
    }
    
    runner = ValidateH5adRunner(args=[], input_data=config)
    runner.run()

    output_manifest = json.load(open(output_json, 'rb'))
    result_path = output_manifest['valid_h5ad_path']

    assert result_path == orig_path

    # make sure input file did not change
    md51 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md51.update(src.read())

    assert md50.hexdigest() == md51.hexdigest()


def test_validation_cli_of_h5ad_missing_output(
        var_fixture,
        obs_fixture,
        x_fixture,
        tmp_dir_fixture):
    """
    Test that an error is raised if output_json is not specified
    """

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    config = {
        'h5ad_path': orig_path,
        'output_dir': str(tmp_dir_fixture.resolve().absolute()),
        'tmp_dir': str(tmp_dir_fixture.resolve().absolute()),
    }

    with pytest.raises(RuntimeError,
                       match="must specify a path for output_json"):
        ValidateH5adRunner(args=[], input_data=config)
