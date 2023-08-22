import pytest

import anndata
import h5py
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.utils.h5_utils import (
    copy_h5_excluding_data)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):

    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('h5_utils'))

    yield tmp_dir

    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def data_fixture():
    rng = np.random.default_rng(876123)

    result = dict()
    result['a'] = rng.random(17)
    result['b'] = 'thequickbrownfox'.encode('utf-8')
    result['c/d'] = rng.random(62)
    result['c/e'] = 'jumpedoverthelazydog'.encode('utf-8')
    result['f/g/h'] = 'youknowhatIamsaing'.encode('utf-8')
    result['f/g/i'] = rng.random(7)
    result['f/j/k'] = rng.random(19)
    result['f/j/l'] = 'seeyoulater'.encode('utf-8')

    return result


@pytest.fixture(scope='module')
def h5_fixture(
        tmp_dir_fixture,
        data_fixture):
    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='baseline_',
        suffix='.h5')
    with h5py.File(h5_path, 'w') as dst:
        for name in data_fixture:
            if name == 'c/d':
                chunks = (12,)
            else:
                chunks = None
            dst.create_dataset(
                name,
                data=data_fixture[name],
                chunks=chunks)

    return h5_path


@pytest.mark.parametrize(
    'excluded_groups,excluded_datasets',
    [
     (None, None),
     (None, ['b']),
     (None, ['f/g/i']),
     (None, ['b', 'f/g/i']),
     (['f'], None),
     (['c', 'f/j'], None),
     (['c', 'f/j'], ['b'])
    ])
def test_h5_copy_util(
        tmp_dir_fixture,
        data_fixture,
        h5_fixture,
        excluded_groups,
        excluded_datasets):

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='dst_file_',
        suffix='.h5')

    copy_h5_excluding_data(
        src_path=h5_fixture,
        dst_path=dst_path,
        excluded_datasets=excluded_datasets,
        excluded_groups=excluded_groups)

    # make sure src file was unchanged
    with h5py.File(h5_fixture, 'r') as src:
        for name in data_fixture:
            if isinstance(data_fixture[name], np.ndarray):
                np.testing.assert_allclose(
                    data_fixture[name],
                    src[name][()],
                    atol=0.0,
                    rtol=1.0e-6)
            else:
                assert data_fixture[name] == src[name][()]

    to_exclude =[]
    if excluded_datasets is not None:
        to_exclude += list(excluded_datasets)
    if excluded_groups is not None:
        to_exclude += list(excluded_groups)

    # test contents of dst file
    with h5py.File(dst_path, 'r') as src:
        for name in data_fixture:
            is_excluded = False
            for bad in to_exclude:
                if bad in name:
                    is_excluded = True

            if is_excluded:
                assert name not in src
            else:
               if isinstance(data_fixture[name], np.ndarray):
                   np.testing.assert_allclose(
                       data_fixture[name],
                       src[name][()],
                       atol=0.0,
                       rtol=1.0e-6)
               else:
                   assert data_fixture[name] == src[name][()]



@pytest.mark.parametrize(
    "excluded_groups",
    [[], ['X'], ['obsm'], ['var', 'varm'], ['varm'], ['obsm'], ['uns'],
     ['X', 'uns'], ['obsm', 'uns'], ['X', 'obsm'], ['varm', 'uns']])
def test_h5_copy_util_on_h5ad(
        tmp_dir_fixture,
        excluded_groups):

    src_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='src_',
        suffix='.h5ad')

    rng = np.random.default_rng(87123)
    n_cells = 151
    n_genes = 431
    n_tot =n_cells*n_genes
    data = np.zeros(n_tot, dtype=np.float32)
    chosen = rng.choice(np.arange(n_tot, dtype=int), n_tot//3, replace=False)
    data[chosen] = rng.random(len(chosen)).astype(np.float32)
    data = data.reshape((n_cells, n_genes))
    x = scipy_sparse.csr_matrix(data)

    obs_data = [
        {'cell_id': f'cell_{ii}',
         'garbage': ii,
         'type': rng.choice(['a', 'b', 'c'])}
        for ii in range(n_cells)
    ]

    obs = pd.DataFrame(obs_data).set_index('cell_id')
    obs['type'] = obs['type'].astype('category')

    var_data = [
        {'gene_id': f'gene_{ii}',
         'junk': ii**2,
         'flavor': rng.choice(['d', 'e', 'f'])}
        for ii in range(n_genes)
    ]

    var = pd.DataFrame(var_data).set_index('gene_id')
    var['flavor'] = var['flavor'].astype('category')

    obsm = {
        'cell_mask': rng.integers(0, 2, n_cells).astype(bool),
        'names': np.array([f'other_name_{ii}' for ii in range(n_cells)])
    }

    varm = {
        'gene_mask': rng.integers(0, 2, n_genes).astype(bool),
        'gene_df': pd.DataFrame([{'gene_id': f'gene_{ii}',
                                  'value': f'something_{ii}',
                                  'cube': ii**3}
                                 for ii in range(n_genes)]).set_index('gene_id')
    }

    uns = {'date': 'today', 'author': 'me'}

    src_a_data = anndata.AnnData(
        X=x,
        obs=obs,
        var=var,
        uns=uns,
        obsm=obsm,
        varm=varm)

    src_a_data.write_h5ad(src_path)

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='dst_',
        suffix='.h5ad')

    copy_h5_excluding_data(
        src_path=src_path,
        dst_path=dst_path,
        tmp_dir=tmp_dir_fixture,
        excluded_groups=excluded_groups,
        excluded_datasets=excluded_groups)


    dst_a_data = anndata.read_h5ad(dst_path, backed='r')
    if 'X' not in excluded_groups:
        dst_X = dst_a_data.X[()].toarray()
        np.testing.assert_allclose(
            dst_X,
            x.toarray(),
            atol=0.0,
            rtol=1.0e-6)
    else:
        with pytest.raises(KeyError, match="object \'X\' doesn\'t exist"):
            dst_X = dst_a_data.X[()]

    dst_obs = dst_a_data.obs
    pd.testing.assert_frame_equal(obs, dst_obs)

    if 'var' not in excluded_groups:
        dst_var = dst_a_data.var
        pd.testing.assert_frame_equal(var, dst_var)
    else:
        dst_var = dst_a_data.var
        assert not var.equals(dst_var)

    if 'obsm' not in excluded_groups:
        dst_obsm = dst_a_data.obsm
        assert set(dst_obsm.keys()) == set(obsm.keys())
        for k in dst_obsm:
            actual = dst_obsm[k]
            expected = obsm[k]
            if isinstance(actual, pd.DataFrame):
                pd.testing.assert_frame_equal(expected, actual)
            elif isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(expected, actual)
            else:
                assert expected == actual
    else:
        dst_obsm = dst_a_data.obsm
        assert dst_obsm == dict()


    if 'varm' not in excluded_groups:
        dst_varm = dst_a_data.varm
        assert set(dst_varm.keys()) == set(varm.keys())
        for k in dst_varm:
            actual = dst_varm[k]
            expected = varm[k]
            if isinstance(actual, pd.DataFrame):
                pd.testing.assert_frame_equal(expected, actual)
            elif isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(expected, actual)
            else:
                assert expected == actual
    else:
        dst_varm = dst_a_data.varm
        assert dst_varm == dict()

    if 'uns' not in excluded_groups:
        dst_uns = dst_a_data.uns
        assert set(dst_uns.keys()) == set(uns.keys())
        for k in dst_uns:
            actual = dst_uns[k]
            expected = uns[k]
            if isinstance(actual, pd.DataFrame):
                pd.testing.assert_frame_equal(expected, actual)
            else:
                assert expected == actual
    else:
        dst_uns = dst_a_data.uns
        assert dst_uns == dict()
