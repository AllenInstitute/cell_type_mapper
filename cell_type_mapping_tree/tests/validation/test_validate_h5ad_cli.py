import pytest

import anndata
import hashlib
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import re
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
        "density,as_layer,round_to_int",
        itertools.product(("csr", "csc", "array"), (True, False), (True, False)))
def test_validation_cli_of_h5ad(
        var_fixture,
        obs_fixture,
        x_fixture,
        tmp_dir_fixture,
        density,
        as_layer,
        round_to_int):

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    if density == "array":
        data = x_fixture
    elif density == "csr":
        data = scipy_sparse.csr_matrix(x_fixture)
    elif density == "csc":
        data = scipy_sparse.csc_matrix(x_fixture)
    else:
        raise RuntimeError(f"unknown density {density}")

    if as_layer:
        x = np.zeros(x_fixture.shape)
        layers = {'garbage': data}
        layer = 'garbage'
    else:
        x = data
        layers = None
        layer = 'X'

    a_data = anndata.AnnData(
        X=x,
        var=var_fixture,
        obs=obs_fixture,
        layers=layers)
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
        'output_json': output_json,
        'layer': layer,
        'round_to_int': round_to_int
    }

    runner = ValidateH5adRunner(args=[], input_data=config)
    runner.run()

    output_manifest = json.load(open(output_json, 'rb'))
    result_path = output_manifest['valid_h5ad_path']

    orig_path = pathlib.Path(orig_path)
    result_path = pathlib.Path(result_path)

    name_parts = result_path.name.split('_')
    int_pattern = re.compile('[0-9]+')
    timestamp = int_pattern.findall(name_parts[-1])[0]
    base_name = orig_path.name.replace('.h5ad', '')
    expected_name = f'{base_name}_VALIDATED_{timestamp}.h5ad'
    assert result_path.name == expected_name

    if round_to_int:
        with h5py.File(result_path, 'r') as in_file:
            if density != 'array':
                data_key = 'X/data'
            else:
                data_key = 'X'
            assert in_file[data_key].dtype == np.uint8

    actual = anndata.read_h5ad(result_path, backed='r')
    pd.testing.assert_frame_equal(obs_fixture, actual.obs)
    actual_x = actual.X

    if density != "array":

        # Sometimes X comes out as a scipy.sparse matrix,
        # sometimes it is an anndata SparseDataset. Unclear
        # why.
        if not hasattr(actual_x, 'toarray'):
           actual_x = actual_x[()]
        actual_x = actual_x.toarray()

        # check that we can at least read back the
        # sparse matrix using scipy library
        if density == "csr":
            scipy_type = scipy_sparse.csr_matrix
        else:
            scipy_type = scipy_sparse.csc_matrix
        with h5py.File(result_path, "r") as src:
            brute_x = scipy_type(
                (src['X/data'][()],
                 src['X/indices'][()],
                 src['X/indptr'][()]),
                shape=x_fixture.shape)
        brute_x = brute_x.toarray()

    else:
        brute_x = None

    if round_to_int:
        assert not np.allclose(actual_x, x_fixture)
        np.testing.assert_array_equal(
            actual_x,
            np.round(x_fixture).astype(int))

        if brute_x is not None:
            np.testing.assert_array_equal(
                brute_x,
                np.round(x_fixture).astype(int))
    else:
        np.testing.assert_allclose(
            actual_x,
            x_fixture,
            atol=0.0,
            rtol=10.e-6)

        if brute_x is not None:
            np.testing.assert_allclose(
                brute_x,
                x_fixture,
                atol=0.0,
                rtol=1.0e-7)

    actual_var = actual.var
    assert len(actual_var.columns) == 2
    assert list(actual_var['gene_id'].values) == list(var_fixture.index.values)
    assert list(actual_var['val'].values) == list(var_fixture.val.values)
    actual_idx = list(actual_var.index.values)
    assert len(actual_idx) == 4
    assert actual_idx[0] == "ENSMUSG00000000318"
    assert actual_idx[1] == "ENSMUSG00000000320"
    assert actual_idx[2] == "ENSMUSG00000000326"
    assert "unmapped" in actual_idx[3]

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


@pytest.mark.parametrize(
        "density", ("csr", "csc", "array"))
def test_validation_cli_of_h5ad_preserve_norm(
        good_var_fixture,
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
        'output_json': output_json,
        'round_to_int': False
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


@pytest.mark.parametrize(
        "density,preserve_norm",
        itertools.product(("csr", "csc", "array"), (True, False)))
def test_validation_cli_of_good_h5ad_in_layer(
        good_var_fixture,
        obs_fixture,
        good_x_fixture,
        x_fixture,
        tmp_dir_fixture,
        density,
        preserve_norm):
    """
    Test behavior of validation CLI when the data is good
    but it is in a non-X layer.
    """

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    if preserve_norm:
        baseline_data = x_fixture
    else:
        baseline_data = good_x_fixture

    if density == "array":
        data = baseline_data
    elif density == "csr":
        data = scipy_sparse.csr_matrix(baseline_data)
    elif density == "csc":
        data = scipy_sparse.csc_matrix(baseline_data)
    else:
        raise RuntimeError(f"unknown density {density}")

    a_data = anndata.AnnData(
        X=np.zeros(data.shape),
        var=good_var_fixture,
        obs=obs_fixture,
        layers={'garbage': data})

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
        'output_json': output_json,
        'layer': 'garbage',
        'round_to_int': not preserve_norm
    }

    runner = ValidateH5adRunner(args=[], input_data=config)
    runner.run()

    output_manifest = json.load(open(output_json, 'rb'))
    result_path = output_manifest['valid_h5ad_path']

    assert result_path != orig_path

    actual = anndata.read_h5ad(result_path)
    pd.testing.assert_frame_equal(
        actual.var,
        good_var_fixture)
    pd.testing.assert_frame_equal(
        actual.obs,
        obs_fixture)

    actual_x = actual.X

    if density != "array":

        # Sometimes X comes out as a scipy.sparse matrix,
        # sometimes it is an anndata SparseDataset. Unclear
        # why.
        if not hasattr(actual_x, 'toarray'):
           actual_x = actual_x[()]
        actual_x = actual_x.toarray()

        # check that we can at least read back the
        # sparse matrix using scipy library
        if density == "csr":
            scipy_type = scipy_sparse.csr_matrix
        else:
            scipy_type = scipy_sparse.csc_matrix
        with h5py.File(result_path, "r") as src:
            brute_x = scipy_type(
                (src['X/data'][()],
                 src['X/indices'][()],
                 src['X/indptr'][()]),
                shape=data.shape)
        brute_x = brute_x.toarray()

    else:
        brute_x = None

    np.testing.assert_allclose(
        actual_x,
        baseline_data,
        atol=0.0,
        rtol=1.0e-6)

    if brute_x is not None:

        np.testing.assert_allclose(
            brute_x,
            baseline_data,
            atol=0.0,
            rtol=1.0e-6)

    # make sure input file did not change
    md51 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md51.update(src.read())

    assert md50.hexdigest() == md51.hexdigest()
