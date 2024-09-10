import pytest

import anndata
import h5py
import itertools
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import shutil

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad,
    copy_layer_to_x,
    read_uns_from_h5ad,
    write_uns_to_h5ad,
    append_to_obsm,
    does_obsm_have_key,
    update_uns,
    amalgamate_csr_to_x,
    amalgamate_dense_to_x,
    shuffle_csr_h5ad_rows,
    pivot_csr_h5ad,
    subset_csc_h5ad_columns)


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

    other_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    shutil.copy(
        src=h5ad_path,
        dst=other_path)

    # test case when 'uns' does not exist
    with h5py.File(other_path, 'a') as dst:
        del dst['uns']

    actual = read_uns_from_h5ad(other_path)
    assert actual == dict()


@pytest.mark.parametrize(
     'which_test', ['basic', 'error', 'clobber'])
def test_update_uns(tmp_dir_fixture, which_test):

    original_uns = {'a': 1, 'b': [1, 2, 3]}
    a_data = anndata.AnnData(
        uns=original_uns)
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='update_uns_',
        suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)

    if which_test == 'basic':
        new_uns = {'c': 'abcd', 'd': 5}
        update_uns(h5ad_path, new_uns=new_uns)

        actual = anndata.read_h5ad(h5ad_path, backed='r')
        assert len(actual.uns) == len(new_uns) + len(original_uns)

        for k in original_uns:
            if isinstance(actual.uns[k], np.ndarray):
                np.testing.assert_array_equal(actual.uns[k], original_uns[k])
            else:
                assert actual.uns[k] == original_uns[k]
        for k in new_uns:
            assert actual.uns[k] == new_uns[k]

    elif which_test == 'error':
        with pytest.raises(RuntimeError, match="key already exists"):
            update_uns(h5ad_path, new_uns={'a':2, 'f': 6}, clobber=False)

        # make sure uns was unchanged
        actual = anndata.read_h5ad(h5ad_path)
        assert set(actual.uns.keys()) == set(original_uns.keys())
        for k in original_uns:
            if isinstance(actual.uns[k], np.ndarray):
                np.testing.assert_array_equal(actual.uns[k], original_uns[k])
            else:
                assert actual.uns[k] == original_uns[k]

    elif which_test == 'clobber':
        update_uns(h5ad_path, new_uns={'a': 2, 'f': 6}, clobber=True)
        actual = anndata.read_h5ad(h5ad_path, backed='r')
        assert len(actual.uns) == 3
        assert actual.uns['a'] == 2
        assert actual.uns['f'] == 6
        np.testing.assert_array_equal(
            actual.uns['b'], original_uns['b'])
    else:
        raise RuntimeError(f"cannot parse which_test = {which_test}")


@pytest.mark.parametrize(
    'clobber', [True, ]
)
def test_compound_update_uns(tmp_dir_fixture, clobber):

    original_uns = {
        'a': 1,
        'b': 9,
        'c': {
            'd': 4,
            'other_dict': {
                'z': 88,
                'u': 14
            }
        }
    }

    a_data = anndata.AnnData(
        uns=original_uns)
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='update_uns_',
        suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)

    new_uns = {
        'c': {
            'e': 2,
            'f': 3,
            'other_dict': {
                'x': 17,
            },
            'still_another_dict': {
                'y': 55
            }
        }
    }

    update_uns(h5ad_path, new_uns=new_uns, clobber=False)
    roundtrip = anndata.read_h5ad(h5ad_path)
    expected = {
        'a': 1,
        'b': 9,
        'c': {
            'd': 4,
            'e': 2,
            'f': 3,
            'other_dict': {
                'z': 88,
                'x': 17,
                'u': 14
            },
            'still_another_dict': {
                'y': 55
            }
        }
    }
    assert roundtrip.uns == expected

    a_data = anndata.AnnData(
        uns=original_uns)
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='update_uns_',
        suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)

    new_uns = {
        'c': {
            'd': 3,
            'e': 4,
            'still_another_dict': {
                'y': 45
            },
            'other_dict': {
                'x': 66,
                'z': 13
            }
        }
    }

    if clobber:
        update_uns(h5ad_path, new_uns=new_uns, clobber=clobber)
        expected = {
            'a': 1,
            'b': 9,
            'c': {
                'd': 3,
                'e': 4,
                'other_dict': {
                    'z': 13,
                    'x': 66,
                    'u': 14
                },
                'still_another_dict': {
                    'y': 45
                }
            }
        }
        roundtrip = anndata.read_h5ad(h5ad_path)
        if roundtrip.uns != expected:
            import json
            from cell_type_mapper.utils.utils import (
                clean_for_json
            )
            msg = (
                f"{json.dumps(clean_for_json(roundtrip.uns), indent=2)}\n"
                "=======n"
                f"{json.dumps(clean_for_json(expected), indent=2)}\n"
            )
            raise RuntimeError(msg)
    else:
        with pytest.raises(RuntimeError, match="key already exists"):
            update_uns(h5ad_path, new_uns=new_uns, clobber=clobber)
        # make sure uns was unchanged
        actual = anndata.read_h5ad(h5ad_path)
        assert actual.uns == original_uns


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


def test_does_obsm_have_key(tmp_dir_fixture):

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
    assert does_obsm_have_key(h5ad_path, 'a')
    assert does_obsm_have_key(h5ad_path, 'b')
    assert not does_obsm_have_key(h5ad_path, 'x')

    # try without obsm
    a_data = anndata.AnnData(
        X=rng.random((n_obs, n_var)))
    a_data.write_h5ad(h5ad_path)
    assert not does_obsm_have_key(h5ad_path, 'a')
    assert not does_obsm_have_key(h5ad_path, 'b')
    assert not does_obsm_have_key(h5ad_path, 'x')


def test_appending_obsm_to_obs(tmp_dir_fixture):
    """
    Test that, if we are adding a dataframe
    to obsm, an error is raised if that dataframe's
    index is not aligned to the index in obs.
    """
    rng = np.random.default_rng(4321233)
    obs_data = [
        {'a': 'foo', 'x': 1},
        {'a': 'bar', 'x': 3},
        {'a': 'baz', 'x': 4}
    ]
    obs = pd.DataFrame(obs_data).set_index('a')
    a_data = anndata.AnnData(
        X=rng.random((3, 4)),
        obs=obs)
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='dummy_h5ad_',
        suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)

    bad_obsm_data = [
        {'a': 'throat warbler', 'z': 3},
        {'a': 'yacht', 'z': 4},
        {'a': 'mangrove', 'z': 5}
    ]
    bad_obsm = pd.DataFrame(bad_obsm_data).set_index('a')
    with pytest.raises(RuntimeError, match='index values are not the same'):
        append_to_obsm(
            h5ad_path=h5ad_path,
            obsm_key='test',
            obsm_value=bad_obsm)


    reordered_obsm_data = [
        {'d': 'baz', 'z': 3},
        {'d': 'foo', 'z': 4},
        {'d': 'bar', 'z': 5}
    ]
    reordered_obsm = pd.DataFrame(reordered_obsm_data).set_index('d')
    append_to_obsm(
        h5ad_path=h5ad_path,
        obsm_key='test',
        obsm_value=reordered_obsm)

    roundtrip = anndata.read_h5ad(h5ad_path)
    assert list(roundtrip.obsm['test'].z.values) == [4, 5, 3]

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='dummy_h5ad_',
        suffix='.h5ad')
    a_data.write_h5ad(h5ad_path)

    good_obsm_data = [
        {'d': 'foo', 'z': 13},
        {'d': 'bar', 'z': 14},
        {'d': 'baz', 'z': 15}
    ]
    good_obsm = pd.DataFrame(good_obsm_data).set_index('d')
    append_to_obsm(
        h5ad_path=h5ad_path,
        obsm_key='test',
        obsm_value=good_obsm)

    roundtrip = anndata.read_h5ad(h5ad_path)
    roundtrip_obsm = roundtrip.obsm
    assert 'test' in roundtrip_obsm
    assert list(roundtrip_obsm['test'].z.values) == [13, 14, 15]


@pytest.mark.parametrize(
    "data_dtype, layer, density, compression",
    itertools.product(
        [float, int, np.uint16],
        ["X"],
        ["csr", "dense"],
        [True, False]))
def test_amalgamate_csr_to_x(
        data_dtype,
        layer,
        density,
        compression,
        tmp_dir_fixture):
    rng = np.random.default_rng(7112233)
    n_rows = 1000
    n_cols = 231
    n_tot = n_rows*n_cols
    data = np.zeros(n_tot, dtype=data_dtype)
    chosen = rng.choice(np.arange(n_tot), n_tot//3, replace=False)
    if data_dtype == float:
        data[chosen] = rng.random(len(chosen))
    elif data_dtype == int:
        data[chosen] = rng.integers(1, 2**23-1, len(chosen)).astype(data_dtype)
    elif data_dtype == np.uint16:
        data[chosen] = rng.integers(1, 255, len(chosen)).astype(data_dtype)
    else:
        raise RuntimeError(
            f"test not designed for type {data_dtype}")

    data = data.reshape((n_rows, n_cols))

    src_path_list = []

    # size of chunks to save in different .h5 files
    d_row_list = [237, 1, 412, 348, 2]

    i0 = 0
    for d_row in d_row_list:
        i1 = i0 + d_row

        h5_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5')
        if density == "csr":
            this = scipy_sparse.csr_matrix(
                data[i0:i1, :])
            with h5py.File(h5_path, 'w') as dst:
                dst.create_dataset(
                    'data', data=this.data)
                dst.create_dataset(
                    'indices', data=this.indices)
                dst.create_dataset(
                    'indptr', data=this.indptr)
        else:
            with h5py.File(h5_path, 'w') as dst:
                dst.create_dataset('data', data=data[i0:i1, :])

        src_path_list.append(h5_path)
        i0 = i1

    var = pd.DataFrame(
        [{'g': f'g_{ii}'} for ii in range(n_cols)]).set_index('g')
    obs = pd.DataFrame(
        [{'c': f'c_{ii}'} for ii  in range(n_rows)]).set_index('c')
    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    full_csr = scipy_sparse.csr_matrix(data)

    a_data = anndata.AnnData(obs=obs, var=var)

    a_data.write_h5ad(h5ad_path)

    del a_data

    if density == "csr":
        amalgamate_csr_to_x(
            src_path_list=src_path_list,
            dst_path=h5ad_path,
            final_shape=(n_rows, n_cols),
            dst_grp=layer,
            compression=compression)
    else:
        amalgamate_dense_to_x(
            src_path_list=src_path_list,
            dst_path=h5ad_path,
            final_shape=(n_rows, n_cols),
            dst_grp=layer,
            compression=compression)

    # check compression
    if compression:
        if layer == 'X':
            key = 'X'
        else:
            key = f'layers/{layer}'
        if density == 'csr':
            key = f'{key}/data'
        with h5py.File(h5ad_path, 'r') as src:
            assert src[key].compression == 'gzip'
            assert src[key].compression_opts == 4

    round_trip = anndata.read_h5ad(h5ad_path, backed='r')

    if layer == 'X':
        actual = round_trip.X[()]
    else:
        actual = round_trip.layers[layer.replace('layers/','')][()]

    if density == "csr":
        actual = actual.toarray()

    np.testing.assert_allclose(
        actual,
        data,
        atol=0.0,
        rtol=1.0e-6)

    if layer == 'X':

        d_chunk = 431
        iterator = round_trip.chunked_X(d_chunk)
        for chunk in iterator:
            expected = data[chunk[1]:chunk[2], :]
            actual = chunk[0]
            if density == "csr":
                actual = actual.toarray()
            np.testing.assert_allclose(actual, expected)

        for idx_list in ([14, 188, 33],
                         [11, 67, 2, 3],
                         [0, 45, 16],
                         [3, 67, 22, 230]):
            col_idx = round_trip.var.index[idx_list]
            actual = round_trip[:, col_idx].to_memory()
            expected = data[:, idx_list]
            actual_x = actual.chunk_X(np.arange(n_rows))
            np.testing.assert_allclose(
                actual_x,
                expected,
                atol=0.0,
                rtol=1.0e-6)


def test_shuffle_csr_h5ad_rows(
        tmp_dir_fixture):

    rng = np.random.default_rng(211131)
    n_rows = 100
    n_cols = 500
    data = rng.integers(0, 255, (n_rows, n_cols), dtype=np.uint8)

    data[11, :] = 0
    data[n_rows-1, :] = 0
    data[:, -1] = 0

    var = pd.DataFrame(
        [{'g': f'g_{ii}', 'y': rng.integers(0, 9999)}
         for ii in range(n_cols)]).set_index('g')

    obs_data = [
        {'c': f'c_{ii}', 'x': rng.integers(0, 9999)}
        for ii in range(n_rows)
    ]

    obs = pd.DataFrame(obs_data).set_index('c')
    csr = anndata.AnnData(
        X=scipy_sparse.csr_matrix(data),
        obs=obs,
        var=var)
    csr_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='pre_shuffle_csr_',
        suffix='.h5ad')
    csr.write_h5ad(csr_path)

    new_row_order = np.arange(n_rows, dtype=int)
    rng.shuffle(new_row_order)
    shuffled_data = data[new_row_order, :]
    shuffled_obs = [
        obs_data[ii]
        for ii in new_row_order]
    shuffled_obs = pd.DataFrame(shuffled_obs).set_index('c')
    shuffled_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='shuffled_csr_',
        suffix='.h5ad')
    shuffled = anndata.AnnData(
        obs=shuffled_obs,
        var=var,
        X=scipy_sparse.csr_matrix(shuffled_data))
    shuffled.write_h5ad(shuffled_path)

    test_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='actual_shuffled_csr_',
        suffix='.h5ad')
    shuffle_csr_h5ad_rows(
        src_path=csr_path,
        dst_path=test_path,
        new_row_order=new_row_order)

    with h5py.File(shuffled_path, 'r') as expected:
        with h5py.File(test_path, 'r') as actual:
            expected_attrs = dict(expected['X'].attrs)
            actual_attrs = dict(actual['X'].attrs)
            for k in expected_attrs:
                if k == 'shape':
                    np.testing.assert_array_equal(
                        actual_attrs[k], expected_attrs[k])
                else:
                    assert actual_attrs[k] == expected_attrs[k]
            for k in ('indices', 'data', 'indptr'):
                np.testing.assert_array_equal(
                    expected[f'X/{k}'][()],
                    actual[f'X/{k}'][()])

    expected = anndata.read_h5ad(shuffled_path, backed='r')
    actual = anndata.read_h5ad(test_path, backed='r')
    pd.testing.assert_frame_equal(expected.var, actual.var)
    pd.testing.assert_frame_equal(expected.obs, actual.obs)


def test_pivot_csr_h5ad(
        tmp_dir_fixture):

    rng = np.random.default_rng(211131)
    n_rows = 100
    n_cols = 500
    data = rng.integers(0, 255, (n_rows, n_cols), dtype=np.uint8)

    data[:, -1] = 0

    var = pd.DataFrame(
        [{'g': f'g_{ii}', 'y': rng.integers(0, 9999)}
         for ii in range(n_cols)]).set_index('g')

    obs_data = [
        {'c': f'c_{ii}', 'x': rng.integers(0, 9999)}
        for ii in range(n_rows)
    ]

    obs = pd.DataFrame(obs_data).set_index('c')
    csr = anndata.AnnData(
        X=scipy_sparse.csr_matrix(data),
        obs=obs,
        var=var)
    csr_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='pre_pivot_csr_',
        suffix='.h5ad')
    csr.write_h5ad(csr_path)

    csc = anndata.AnnData(
        X=scipy_sparse.csc_matrix(data),
        obs=obs,
        var=var)
    csc_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='expected_csc_',
        suffix='.h5ad')
    csc.write_h5ad(csc_path)

    test_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='pivoted_',
        suffix='.h5ad')

    pivot_csr_h5ad(
        src_path=csr_path,
        dst_path=test_path,
        tmp_dir=tmp_dir_fixture,
        max_gb=1)

    with h5py.File(csc_path, 'r') as expected:
        with h5py.File(test_path, 'r') as actual:
            expected_attrs = dict(expected['X'].attrs)
            actual_attrs = dict(actual['X'].attrs)
            for k in expected_attrs:
                if k == 'shape':
                    np.testing.assert_array_equal(
                        actual_attrs[k], expected_attrs[k])
                else:
                    assert actual_attrs[k] == expected_attrs[k]
            for k in ('indices', 'data', 'indptr'):
                np.testing.assert_array_equal(
                    expected[f'X/{k}'][()],
                    actual[f'X/{k}'][()])

    expected = anndata.read_h5ad(csc_path, backed='r')
    actual = anndata.read_h5ad(test_path, backed='r')
    pd.testing.assert_frame_equal(expected.var, actual.var)
    pd.testing.assert_frame_equal(expected.obs, actual.obs)


def test_subset_csc_h5ad_columns(
        tmp_dir_fixture):
    rng = np.random.default_rng(761231)
    n_rows = 100
    n_cols = 500
    data = rng.integers(0, 255, (n_rows, n_cols), dtype=np.uint8)
    chosen_cols = rng.choice(np.arange(n_cols), 77, replace=False)
    data[:, chosen_cols[2]] = 0

    obs = pd.DataFrame(
        [{'c': f'c_{ii}', 'x': rng.integers(0, 9999)}
         for ii in range(n_rows)]).set_index('c')

    var_data = [
        {'g': f'g_{ii}', 'y': rng.integers(0, 9999)}
        for ii in range(n_cols)]

    var = pd.DataFrame(
        var_data).set_index('g')

    src_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='subset_col_src_',
        suffix='.h5ad')

    src = anndata.AnnData(
        obs=obs,
        var=var,
        X=scipy_sparse.csc_matrix(data))
    src.write_h5ad(src_path)

    sorted_cols = np.sort(chosen_cols)
    assert not np.array_equal(chosen_cols, sorted_cols)

    expected_data = data[:, sorted_cols]
    expected_var = pd.DataFrame(
        [var_data[ii] for ii in sorted_cols]).set_index('g')
    expected_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='subset_col_expected_',
        suffix='.h5ad')
    expected = anndata.AnnData(
        obs=obs,
        var=expected_var,
        X=scipy_sparse.csc_matrix(expected_data))
    expected.write_h5ad(expected_path)

    test_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='subset_col_test_',
        suffix='.h5ad')

    subset_csc_h5ad_columns(
        src_path=src_path,
        dst_path=test_path,
        chosen_columns=chosen_cols)

    with h5py.File(expected_path, 'r') as expected:
        with h5py.File(test_path, 'r') as actual:
            expected_attrs = dict(expected['X'].attrs)
            actual_attrs = dict(actual['X'].attrs)
            for k in expected_attrs:
                if k == 'shape':
                    np.testing.assert_array_equal(
                        actual_attrs[k], expected_attrs[k])
                else:
                    assert actual_attrs[k] == expected_attrs[k]
            for k in ('indices', 'data', 'indptr'):
                np.testing.assert_array_equal(
                    expected[f'X/{k}'][()],
                    actual[f'X/{k}'][()])

    expected = anndata.read_h5ad(expected_path, backed='r')
    actual = anndata.read_h5ad(test_path, backed='r')
    pd.testing.assert_frame_equal(expected.var, actual.var)
    pd.testing.assert_frame_equal(expected.obs, actual.obs)
