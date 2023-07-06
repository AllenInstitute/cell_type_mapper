import pytest

import anndata
import hashlib
import h5py
import itertools
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.gene_id.gene_id_mapper import (
    GeneIdMapper)

from hierarchical_mapping.validation.validate_h5ad import (
    validate_h5ad)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('validating_h5ad_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def map_data_fixture():

    data = {
        "gene_0": {
            "name": "alice",
            "nickname": "allie"
        },
        "gene_1": {
            "name": "robert"
        },
        "gene_2": {
            "nickname": "hammer"
        },
        "gene_3": {
            "name": "charlie",
            "nickname": "chuck"
        }
    }

    return data

@pytest.fixture
def var_fixture():
    records = [
        {'gene_id': 'robert', 'val': 'ab'},
        {'gene_id': 'tyler', 'val': 'cd'},
        {'gene_id': 'alice', 'val': 'xy'},
        {'gene_id': 'hammer', 'val': 'uw'}]

    return pd.DataFrame(records).set_index('gene_id')


@pytest.fixture
def good_var_fixture():
    records = [
        {'gene_id': 'gene_2', 'val': 'ab'},
        {'gene_id': 'gene_3', 'val': 'cd'},
        {'gene_id': 'gene_1', 'val': 'xy'},
        {'gene_id': 'gene_0', 'val': 'uw'}]

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
        itertools.product(
         ("csr", "csc", "array"),
         (True, False),
         (True, False)))
def test_validation_of_h5ad(
        var_fixture,
        obs_fixture,
        x_fixture,
        map_data_fixture,
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
    else:
        x = data
        layers = None

    a_data = anndata.AnnData(
        X=x,
        var=var_fixture,
        obs=obs_fixture,
        layers=layers)
    a_data.write_h5ad(orig_path)

    md50 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md50.update(src.read())

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    if as_layer:
        layer = 'garbage'
    else:
        layer = 'X'

    result_path, _ = validate_h5ad(
        h5ad_path=orig_path,
        output_dir=tmp_dir_fixture,
        gene_id_mapper=gene_id_mapper,
        tmp_dir=tmp_dir_fixture,
        layer=layer,
        round_to_int=round_to_int)

    assert result_path is not None

    if round_to_int:
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

    if round_to_int:
        assert not np.allclose(actual_x, x_fixture)
        assert np.array_equal(
            actual_x,
            np.round(x_fixture).astype(int))
    else:
        assert np.allclose(
            actual_x,
            x_fixture,
            atol=0.0,
            rtol=1.0e-6)

    actual_var = actual.var
    assert len(actual_var.columns) == 2
    assert list(actual_var['gene_id'].values) == list(var_fixture.index.values)
    assert list(actual_var['val'].values) == list(var_fixture.val.values)
    actual_idx = list(actual_var.index.values)
    assert len(actual_idx) == 4
    assert actual_idx[0] == "gene_1"
    assert "nonsense" in actual_idx[1]
    assert actual_idx[2] == "gene_0"
    assert "nonsense" in actual_idx[3]

    # make sure input file did not change
    md51 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md51.update(src.read())

    assert md50.hexdigest() == md51.hexdigest()

    # test that gene ID mapping is in unstructured
    # metadata
    uns = actual.uns
    assert 'AIBS_CDM_gene_mapping' in uns

    old_genes = list(var_fixture.index.values)
    new_genes = list(actual.var.index.values)
    for old, new in zip(old_genes, new_genes):
        assert uns['AIBS_CDM_gene_mapping'][old] == new
    assert len(uns['AIBS_CDM_gene_mapping']) == len(old_genes)


@pytest.mark.parametrize(
        "density", ("csr", "csc", "array"))
def test_validation_of_good_h5ad(
        good_var_fixture,
        obs_fixture,
        good_x_fixture,
        map_data_fixture,
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

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    result_path, _ = validate_h5ad(
        h5ad_path=orig_path,
        output_dir=tmp_dir_fixture,
        gene_id_mapper=gene_id_mapper,
        tmp_dir=tmp_dir_fixture)

    assert result_path is None

    # make sure input file did not change
    md51 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md51.update(src.read())

    assert md50.hexdigest() == md51.hexdigest()


@pytest.mark.parametrize(
        "density", ("csr", "csc", "array"))
def test_validation_of_h5ad_ignoring_norm(
        good_var_fixture,
        obs_fixture,
        x_fixture,
        map_data_fixture,
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

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    result_path, _ = validate_h5ad(
        h5ad_path=orig_path,
        output_dir=tmp_dir_fixture,
        gene_id_mapper=gene_id_mapper,
        tmp_dir=tmp_dir_fixture,
        round_to_int=False)

    assert result_path is None

    # make sure input file did not change
    md51 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md51.update(src.read())

    assert md50.hexdigest() == md51.hexdigest()

@pytest.mark.parametrize(
        "density", ("csr", "csc", "array"))
def test_validation_of_good_h5ad_in_layer(
        good_var_fixture,
        obs_fixture,
        good_x_fixture,
        map_data_fixture,
        tmp_dir_fixture,
        density):
    """
    Test that new file is written if otherwise
    good cell by gene data is in non-X layer
    """
    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    if density == "array":
        data = good_x_fixture
    elif density == "csr":
        data = scipy_sparse.csr_matrix(good_x_fixture)
    elif density == "csc":
        data = scipy_sparse.csc_matrix(good_x_fixture)
    else:
        raise RuntimeError(f"unknown density {density}")

    a_data = anndata.AnnData(
        X=np.zeros(good_x_fixture.shape),
        var=good_var_fixture,
        obs=obs_fixture,
        layers={'garbage': data})
    a_data.write_h5ad(orig_path)

    md50 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md50.update(src.read())

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    result_path, _ = validate_h5ad(
        h5ad_path=orig_path,
        output_dir=tmp_dir_fixture,
        gene_id_mapper=gene_id_mapper,
        tmp_dir=tmp_dir_fixture,
        layer='garbage')

    assert result_path is not None

    # make sure input file did not change
    md51 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md51.update(src.read())

    assert md50.hexdigest() == md51.hexdigest()

    actual = anndata.read_h5ad(result_path, backed='r')
    pd.testing.assert_frame_equal(
        good_var_fixture,
        actual.var)

    actual_x = actual.X[()]
    if density != "array":
        actual_x = actual_x.toarray()
    np.testing.assert_allclose(
        good_x_fixture,
        actual_x,
        atol=0.0,
        rtol=1.0e-6)

    # test that gene ID mapping is not in unstructured
    # metadata
    uns = actual.uns
    assert 'AIBS_CDM_gene_mapping' not in uns


@pytest.mark.parametrize(
        "density,output_dtype",
        itertools.product(
            ("csr", "csc", "array"),
            (np.uint8, np.int8, np.uint16, np.int16,
             np.uint32, np.int32, np.uint64, np.int64)))
def test_validation_of_h5ad_diverse_dtypes(
        var_fixture,
        obs_fixture,
        map_data_fixture,
        tmp_dir_fixture,
        density,
        output_dtype):
    """
    Make sure that correct dtype is chosen
    """
    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    output_info = np.iinfo(output_dtype)
    n_rows = len(obs_fixture)
    n_cols = len(var_fixture)
    min_val = float(output_info.min)+0.1
    max_val = float(output_info.max)-0.1
    x_data = np.zeros((n_rows, n_cols), float)
    rng = np.random.default_rng(2231)
    for i_row in range(n_rows):
        chosen = np.unique(rng.integers(0, n_cols, 3))
        for i_col in chosen:
            x_data[i_row, i_col] = 2.0+10.0*rng.random()
    x_data[0, 1] = min_val
    x_data[2, 1] = max_val

    if density == "array":
        x = x_data
    elif density == "csr":
        x = scipy_sparse.csr_matrix(x_data)
    elif density == "csc":
        x = scipy_sparse.csc_matrix(x_data)
    else:
        raise RuntimeError(f"unknown density {density}")

    a_data = anndata.AnnData(
        X=x,
        var=var_fixture,
        obs=obs_fixture,
        dtype=np.float64)
    a_data.write_h5ad(orig_path)

    md50 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md50.update(src.read())

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    result_path, _ = validate_h5ad(
        h5ad_path=orig_path,
        output_dir=tmp_dir_fixture,
        gene_id_mapper=gene_id_mapper,
        tmp_dir=tmp_dir_fixture)

    assert result_path is not None

    with h5py.File(result_path, 'r') as in_file:
        if density != 'array':
            data_key = 'X/data'
        else:
            data_key = 'X'
        assert in_file[data_key].dtype == output_dtype
