import pytest

import anndata
import hashlib
import h5py
import itertools
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
from unittest.mock import patch

from cell_type_mapper.test_utils.anndata_utils import (
    write_anndata_x_to_csv
)

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.gene_id.gene_id_mapper import (
    GeneIdMapper)

from cell_type_mapper.validation.validate_h5ad import (
    validate_h5ad)

from cell_type_mapper.cli.cli_log import (
    CommandLog)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('validating_h5ad_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def map_data_fixture():

    data = {
        "alice": "ENSG0",
        "allie": "ENSG0",
        "robert": "ENSG1",
        "hammer": "ENSG2",
        "charlie": "ENSG3",
        "chuck": "ENSG3"
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
        {'gene_id': 'ENSG2', 'val': 'ab'},
        {'gene_id': 'ENSG3', 'val': 'cd'},
        {'gene_id': 'ENSG1', 'val': 'xy'},
        {'gene_id': 'ENSG0', 'val': 'uw'}]

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


@pytest.mark.parametrize('as_layer, with_log',
    itertools.product([True, False], [True, False]))
def test_validation_of_h5ad_without_encoding(
        var_fixture,
        obs_fixture,
        x_fixture,
        map_data_fixture,
        tmp_dir_fixture,
        as_layer,
        with_log):
    """
    Test that we can validate a file which does not have
    'encoding-type' in its metadata
    """

    if with_log:
        log = CommandLog()
    else:
        log = None

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    data = x_fixture

    a_data = anndata.AnnData(
        var=var_fixture,
        obs=obs_fixture)
    a_data.write_h5ad(orig_path)


    if as_layer:
        layer = 'garbage'
        to_write = f'layers/{layer}'
    else:
        to_write = 'X'
        layer = 'X'

    with h5py.File(orig_path, 'a') as dst:
        dst.create_dataset(to_write, data=x_fixture)

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    valid_h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    validate_h5ad(
        h5ad_path=orig_path,
        output_dir=None,
        valid_h5ad_path=valid_h5ad_path,
        gene_id_mapper=gene_id_mapper,
        tmp_dir=tmp_dir_fixture,
        layer=layer,
        round_to_int=True,
        log=log)

@pytest.mark.parametrize('with_log',
    [True, False])
def test_validation_of_corrupted_h5ad(
        var_fixture,
        obs_fixture,
        x_fixture,
        map_data_fixture,
        tmp_dir_fixture,
        with_log):
    """
    Test that the correct failure message is emitted when the h5ad file
    is so corrupted h5py cannot open it
    """

    if with_log:
        log = CommandLog()
    else:
        log = None

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix='.h5ad')

    a_data = anndata.AnnData(
        var=var_fixture,
        obs=obs_fixture,
        X=x_fixture)
    a_data.write_h5ad(orig_path)

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    valid_h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    def bad_file(*args, **kwargs):
        raise RuntimeError("not going to open this")
    with patch('h5py.File', bad_file):
        with pytest.raises(RuntimeError, match="could not even be opened"):
            validate_h5ad(
                h5ad_path=orig_path,
                output_dir=None,
                valid_h5ad_path=valid_h5ad_path,
                gene_id_mapper=gene_id_mapper,
                tmp_dir=tmp_dir_fixture,
                layer='X',
                round_to_int=True,
                log=log)


@pytest.mark.parametrize(
        "density,as_layer,round_to_int,specify_path",
        itertools.product(
         ("csr", "csc", "array", "csv", "gz"),
         (True, False),
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
        round_to_int,
        specify_path):

    if density in ("csv", "gz"):
        if density == "csv":
            suffix = ".csv"
        else:
            suffix = ".csv.gz"

        if as_layer:
            return
    else:
        suffix = ".h5ad"

    orig_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='orig_',
        suffix=suffix)

    if density in ("array", "csv", "gz"):
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

    if density in ("csv", "gz"):
        write_anndata_x_to_csv(
            anndata_obj=a_data,
            dst_path=orig_path
        )
    else:
        a_data.write_h5ad(orig_path)

    md50 = hashlib.md5()
    with open(orig_path, 'rb') as src:
        md50.update(src.read())

    gene_id_mapper = GeneIdMapper(data=map_data_fixture)

    if as_layer:
        layer = 'garbage'
    else:
        layer = 'X'

    if specify_path:
        output_dir = None
        valid_h5ad_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad')
    else:
        output_dir = tmp_dir_fixture
        valid_h5ad_path = None

    result_path, _ = validate_h5ad(
        h5ad_path=orig_path,
        output_dir=output_dir,
        valid_h5ad_path=valid_h5ad_path,
        gene_id_mapper=gene_id_mapper,
        tmp_dir=tmp_dir_fixture,
        layer=layer,
        round_to_int=round_to_int)

    assert result_path is not None
    if specify_path:
        assert str(result_path.resolve().absolute()) == valid_h5ad_path

    if round_to_int:
        with h5py.File(result_path, 'r') as in_file:
            if density not in ('array', 'csv', 'gz'):
                data_key = 'X/data'
            else:
                data_key = 'X'
            assert in_file[data_key].dtype == np.uint8

    actual = anndata.read_h5ad(result_path, backed='r')

    if density not in ("csv", "gz"):
        pd.testing.assert_frame_equal(obs_fixture, actual.obs)
    else:
        np.testing.assert_array_equal(
            obs_fixture.index.values,
            actual.obs.index.values
        )

    actual_x = actual.X[()]
    if density not in ("array", "csv", "gz"):
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

    if density not in ("csv", "gz"):
        assert len(actual_var.columns) == 2
        assert list(actual_var['gene_id'].values) == list(var_fixture.index.values)
        assert list(actual_var['val'].values) == list(var_fixture.val.values)
    actual_idx = list(actual_var.index.values)
    assert len(actual_idx) == 4
    assert actual_idx[0] == "ENSG1"
    assert "unmapped" in actual_idx[1]
    assert actual_idx[2] == "ENSG0"
    assert actual_idx[3] == "ENSG2"

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


def test_validation_of_h5ad_errors():
    """
    Check that you cannot specify both valid_h5ad_path and output_dir
    """
    with pytest.raises(RuntimeError, match="Cannot specify both"):
        validate_h5ad(
            h5ad_path='silly',
            gene_id_mapper='nonsense',
            tmp_dir=None,
            output_dir='foo',
            valid_h5ad_path='bar')

    with pytest.raises(RuntimeError, match="Must specify one of either"):
        validate_h5ad(
            h5ad_path='silly',
            gene_id_mapper='nonsense',
            tmp_dir=None,
            output_dir=None,
            valid_h5ad_path=None)


def test_gene_name_errors(tmp_dir_fixture):
    """
    Test that an error is raised if a gene name is repeated or if a gene
    name is empty (that last one causes an inscrutable error in anndata
    when writing uns to the validated file, if you are not careful)
    """
    n_cells = 5
    obs = pd.DataFrame(
        [{'cell_id': f'c_{ii}'}
         for ii in range(n_cells)]).set_index('cell_id')

    # case of null gene name
    var = pd.DataFrame(
        [{'gene_id': 'a'},
         {'gene_id': ''},
         {'gene_id': 'b'}]).set_index('gene_id')

    a = anndata.AnnData(
        X=np.random.random_sample((n_cells, len(var))),
        var=var,
        obs=obs)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a.write_h5ad(h5ad_path)
    with pytest.raises(RuntimeError, match="gene name '' is invalid"):
        validate_h5ad(
            h5ad_path=h5ad_path,
            valid_h5ad_path=mkstemp_clean(dir=tmp_dir_fixture),
            gene_id_mapper=None)

    # case of repeated gene name
    var = pd.DataFrame(
        [{'gene_id': 'a'},
         {'gene_id': 'b'},
         {'gene_id': 'b'}]).set_index('gene_id')

    a = anndata.AnnData(
        X=np.random.random_sample((n_cells, len(var))),
        var=var,
        obs=obs)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a.write_h5ad(h5ad_path)
    with pytest.raises(RuntimeError, match="gene names must be unique"):
        validate_h5ad(
            h5ad_path=h5ad_path,
            valid_h5ad_path=mkstemp_clean(dir=tmp_dir_fixture),
            gene_id_mapper=None)

    # test that an error is raised if, after clipping the
    # suffix from the Ensembl ID, the list of genes is not
    # unique
    var = pd.DataFrame(
        [{'gene_id': 'ENSG778.3'},
         {'gene_id': 'ENSF5'},
         {'gene_id': 'ENSG778.9'}]).set_index('gene_id')

    a = anndata.AnnData(
        X=np.random.random_sample((n_cells, len(var))),
        var=var,
        obs=obs)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a.write_h5ad(h5ad_path)
    msg = "mapped to identical gene identifiers"
    with pytest.raises(RuntimeError, match=msg):
        validate_h5ad(
            h5ad_path=h5ad_path,
            valid_h5ad_path=mkstemp_clean(dir=tmp_dir_fixture),
            gene_id_mapper=GeneIdMapper.from_mouse())

    # check that an error is raised if two input genes map to the
    # same gene identifiers
    var = pd.DataFrame(
        [{'gene_id': 'Xkr4'},
         {'gene_id': 'ENSF5'},
         {'gene_id': 'ENSMUSG00000051951'}]).set_index('gene_id')

    a = anndata.AnnData(
        X=np.random.random_sample((n_cells, len(var))),
        var=var,
        obs=obs)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    a.write_h5ad(h5ad_path)
    msg = "mapped to identical gene identifiers"
    with pytest.raises(RuntimeError, match=msg):
        validate_h5ad(
            h5ad_path=h5ad_path,
            valid_h5ad_path=mkstemp_clean(dir=tmp_dir_fixture),
            gene_id_mapper=GeneIdMapper.from_mouse())


def test_cell_id_errors(tmp_dir_fixture):
    """
    Test that an error is raised when a cell_id is
    repeated
    """
    obs = pd.DataFrame(
        [
         {'cell_id': 'c0'},
         {'cell_id': 'c1'},
         {'cell_id': 'c0'},
         {'cell_id': 'c2'}
        ]).set_index('cell_id')

    var = pd.DataFrame(
        [{'g': f'g{ii}'} for ii in range(8)]).set_index('g')

    x = np.ones((len(obs), len(var)), dtype=float)
    a_data = anndata.AnnData(
        obs=obs, var=var, X=x)
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)

    msg = "Cell IDs need to be unique"
    with pytest.raises(RuntimeError, match=msg):
        validate_h5ad(
            h5ad_path=h5ad_path,
            valid_h5ad_path=mkstemp_clean(dir=tmp_dir_fixture),
            gene_id_mapper=None)
