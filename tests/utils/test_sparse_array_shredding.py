"""
Test the utility that creates a single sparse array in a single
HDF5 file from a subset of rows in other sparse arrays stored
in other HDF5 files
"""
import pytest

import anndata
import h5py
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as scipy_sparse
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)

from cell_type_mapper.utils.anndata_manipulation import (
    amalgamate_h5ad,
    amalgamate_h5ad_from_label_list
)


@pytest.mark.parametrize(
    'data_dtype, layer, density, dst_sparse',
    itertools.product(
        [np.uint8, np.uint16, np.int16, float],
        ['X', 'dummy', 'raw/X'],
        ['csr', 'csc', 'dense'],
        [True, False])
)
def test_csr_amalgamation(
        tmp_dir_fixture,
        data_dtype,
        layer,
        density,
        dst_sparse):

    rng = np.random.default_rng(712231)
    n_cols = 15

    if data_dtype != float:
        iinfo = np.iinfo(data_dtype)
        d_max = iinfo.max
        d_min = iinfo.min
        if d_min == 0:
            d_min = 1

    src_rows = []
    expected_rows = []

    for ii in range(4):
        n_rows = rng.integers(10, 20)
        n_tot = n_rows*n_cols
        data = np.zeros(n_tot, dtype=float)
        non_null = rng.choice(
            np.arange(n_tot),
            n_tot//5,
            replace=False)

        if data_dtype == float:
            data[non_null] = rng.random(len(non_null))
        else:
            data[non_null] = rng.integers(
                d_min,
                d_max+1,
                len(non_null)).astype(float)

        data = data.reshape((n_rows, n_cols))
        chosen_rows = np.sort(rng.choice(np.arange(n_rows),
                                         rng.integers(5, 7),
                                         replace=False))

        # make sure some empty rows are included
        data[chosen_rows[1], :] = 0

        for idx in chosen_rows:
            expected_rows.append(data[idx, :])

        this_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad')

        if density == 'csr':
            data = scipy_sparse.csr_matrix(data.astype(data_dtype))
        elif density == 'csc':
            data = scipy_sparse.csc_matrix(data.astype(data_dtype))
        else:
            data = data.astype(data_dtype)

        if layer == 'X':
            x = data
            layers = None
            raw = None
        elif layer == 'dummy':
            x = np.zeros(data.shape, dtype=int)
            layers = {layer: data}
            raw = None
        elif layer == 'raw/X':
            x = np.zeros(data.shape, dtype=int)
            layers = None
            raw = {'X': data}
        else:
            raise RuntimeError(
                f"Test does not know how to parse layer '{layer}'"
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a_data = anndata.AnnData(
                X=x,
                layers=layers,
                raw=raw,
                dtype=data_dtype)

        a_data.write_h5ad(this_path)

        del a_data

        src_rows.append(
            {'path': this_path,
             'rows': list(chosen_rows),
             'layer': layer})

    expected_array = np.stack(expected_rows)

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    dst_var = pd.DataFrame(
        [{'g': f'g_{ii}'}
         for ii in range(n_cols)]).set_index('g')
    dst_obs = pd.DataFrame(
        [{'c': f'c_{ii}'}
         for ii in range(expected_array.shape[0])]).set_index('c')

    amalgamate_h5ad(
        src_rows=src_rows,
        dst_path=dst_path,
        dst_var=dst_var,
        dst_obs=dst_obs,
        dst_sparse=dst_sparse,
        tmp_dir=tmp_dir_fixture)

    if dst_sparse:

        with h5py.File(dst_path, 'r') as dst:
            assert dst['X/indices'].dtype == np.int32
            assert dst['X/indptr'].dtype == np.int32
            actual = scipy_sparse.csr_matrix(
                (dst['X/data'][()],
                 dst['X/indices'][()],
                 dst['X/indptr'][()]),
                shape=expected_array.shape)
            actual = actual.toarray()

    else:
        with h5py.File(dst_path, 'r') as dst:
            actual = dst['X'][()]

    np.testing.assert_allclose(
        actual,
        expected_array)

    assert actual.dtype == data_dtype


@pytest.mark.parametrize(
    'data_dtype, layer, density, dst_sparse',
    itertools.product(
        [np.uint8, np.uint16, np.int16, float],
        ['X', 'dummy', 'raw/X'],
        ['csr', 'csc', 'dense'],
        [True, False])
)
def test_csr_anndata_amalgamation(
        tmp_dir_fixture,
        data_dtype,
        layer,
        density,
        dst_sparse):

    rng = np.random.default_rng(712231)
    n_cols = 15

    if data_dtype != float:
        iinfo = np.iinfo(data_dtype)
        d_max = iinfo.max
        d_min = iinfo.min
        if d_min == 0:
            d_min = 1

    src_rows = []
    expected_rows = []

    for ii in range(4):
        n_rows = rng.integers(10, 20)
        n_tot = n_rows*n_cols
        data = np.zeros(n_tot, dtype=float)
        non_null = rng.choice(
            np.arange(n_tot),
            n_tot//5,
            replace=False)

        if data_dtype == float:
            data[non_null] = rng.random(len(non_null))
        else:
            data[non_null] = rng.integers(
                d_min,
                d_max+1,
                len(non_null)).astype(float)

        data = data.reshape((n_rows, n_cols))
        chosen_rows = np.sort(rng.choice(np.arange(n_rows),
                                         rng.integers(5, 7),
                                         replace=False))

        # make sure some empty rows are included
        data[chosen_rows[1], :] = 0

        for idx in chosen_rows:
            expected_rows.append(data[idx, :])

        this_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad')

        if density == 'csr':
            data = scipy_sparse.csr_matrix(data.astype(data_dtype))
        elif density == 'csc':
            data = scipy_sparse.csc_matrix(data.astype(data_dtype))
        else:
            data = data.astype(data_dtype)

        if layer == 'X':
            x = data
            layers = None
            raw = None
        elif layer == 'dummy':
            x = np.zeros(data.shape, dtype=int)
            layers = {layer: data}
            raw = None
        elif layer == 'raw/X':
            x = np.zeros(data.shape, dtype=int)
            layers = None
            raw = {'X': data}
        else:
            raise RuntimeError(
                f"Test does not know how to parse layer '{layer}'"
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a_data = anndata.AnnData(
                X=x,
                layers=layers,
                raw=raw,
                dtype=data_dtype)

        a_data.write_h5ad(this_path)

        del a_data

        src_rows.append(
            {'path': this_path,
             'rows': list(chosen_rows),
             'layer': layer})

    expected_array = np.stack(expected_rows)

    new_obs = pd.DataFrame(
        [{'cell_id': f'c_{ii}', 'junk': ii**2}
         for ii in range(expected_array.shape[0])]).set_index('cell_id')

    new_var = pd.DataFrame(
        [{'gene': f'g_{ii}', 'garbage': ii**3}
         for ii in range(expected_array.shape[1])]).set_index('gene')

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5ad')

    amalgamate_h5ad(
        src_rows=src_rows,
        dst_path=dst_path,
        dst_obs=new_obs,
        dst_var=new_var,
        dst_sparse=dst_sparse)

    actual_a = anndata.read_h5ad(dst_path, backed='r')
    pd.testing.assert_frame_equal(actual_a.obs, new_obs)
    pd.testing.assert_frame_equal(actual_a.var, new_var)

    if dst_sparse:
        actual_x = actual_a.X[()].todense()
    else:
        actual_x = actual_a.X[()]

    np.testing.assert_allclose(
        actual_x,
        expected_array)

    assert actual_x.dtype == data_dtype

    # test that the resulting anndata object can be sliced on columns
    col_idx = [2, 8, 1, 11]
    col_pd_idx = actual_a.var.index[col_idx]
    actual = actual_a[:, col_pd_idx].to_memory()
    expected = expected_array[:, col_idx]
    np.testing.assert_allclose(
        actual.chunk_X(np.arange(len(actual_a.obs))),
        expected,
        atol=0.0,
        rtol=1.0e-6)


@pytest.mark.parametrize('layer', ['X', 'dummy', 'raw/X'])
def test_failure_when_many_floats(tmp_dir_fixture, layer):
    """
    Test that amalgamation fails when the input arrays
    have disparate float dtypes
    """
    rng = np.random.default_rng(712231)
    n_cols = 15

    src_rows = []
    expected_rows = []

    for ii, data_dtype in enumerate(
                [np.float32, np.float64, np.float32]):
        n_rows = rng.integers(10, 20)
        n_tot = n_rows*n_cols
        data = np.zeros(n_tot, dtype=float)
        non_null = rng.choice(
            np.arange(n_tot),
            n_tot//5,
            replace=False)

        data[non_null] = rng.random(len(non_null), dtype=data_dtype)
        data = data.reshape((n_rows, n_cols))
        chosen_rows = np.sort(rng.choice(np.arange(n_rows),
                                         rng.integers(5, 7),
                                         replace=False))

        # make sure some empty rows are included
        data[chosen_rows[1], :] = 0

        for idx in chosen_rows:
            expected_rows.append(data[idx, :])

        this_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad')

        if layer == 'X':
            x = scipy_sparse.csr_matrix(data)
            layers = None
            raw = None
        elif layer == 'dummy':
            x = np.zeros(data.shape, dtype=int)
            layers = {layer: scipy_sparse.csr_matrix(data.astype(data_dtype))}
            raw = None
        elif layer == 'raw/X':
            x = np.zeros(data.shape, dtype=int)
            layers = None
            raw = {'X': scipy_sparse.csr_matrix(data.astype(data_dtype))}
        else:
            raise RuntimeError(
                f"Test does not know how to parse layer '{layer}'"
            )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            a_data = anndata.AnnData(
                X=x,
                layers=layers,
                raw=raw,
                dtype=data_dtype)

        a_data.write_h5ad(this_path)

        del a_data

        src_rows.append(
            {'path': this_path,
             'rows': list(chosen_rows),
             'layer': layer})

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    with pytest.raises(RuntimeError, match="disparate data types"):
        amalgamate_h5ad(
            src_rows=src_rows,
            dst_path=dst_path,
            dst_obs=None,
            dst_var=None)


@pytest.fixture(scope='module')
def label_to_row_fixture():
    """
    Mapping from cell_label to the row of a cell-by-gene
    matrix (for testing amalgamation from a list of cell_labels)
    """
    rng = np.random.default_rng(22111)
    n_genes = 55
    n_cells = 341
    lookup = dict()
    for i_cell in range(n_cells):
        cell_label = f'cell_{i_cell}'
        row = rng.random(n_genes)
        zeroed_idx = rng.choice(
            np.arange(n_genes),
            rng.integers(5, 45),
            replace=False)
        row[zeroed_idx] = 0.0
        lookup[cell_label] = row
    return lookup


@pytest.fixture(scope='module')
def label_to_obs_fixture(
        label_to_row_fixture):
    """
    Mapping from cell_label to obs metadata associated with
    that cell (for testing amalgamation from a list of cell_labels)
    """
    rng = np.random.default_rng(7712131)

    lookup = dict()
    for i_cell, cell_label in enumerate(label_to_row_fixture):
        this = {
            'cell_label': cell_label,
            'field0': f'f0_{i_cell}',
            'field1': f'a{rng.integers(0,999)}'
        }
        lookup[cell_label] = this
    return lookup


@pytest.fixture(scope='module')
def src_h5ad_list_fixture(
        tmp_dir_fixture,
        label_to_row_fixture,
        label_to_obs_fixture):
    """
    List of h5ad files to use as the source for testing
    amalgamation from a list of cell_labels
    """

    rng = np.random.default_rng(77112211)

    label_list = list(label_to_row_fixture.keys())
    n_cells = len(label_list)
    label_list.sort()
    label_subset_list = [
        label_list[:93],
        label_list[94:171],
        label_list[172:255],
        label_list[256:n_cells]
    ]

    n_genes = len(label_to_row_fixture[label_list[0]])
    var = pd.DataFrame(
        [{'gene_id': f'g_{ii}',
          'other': f'b{rng.integers(5, 9999)}'}
         for ii in range(n_genes)
         ]
    ).set_index('gene_id')

    h5ad_path_list = []
    for label_subset in label_subset_list:
        obs = pd.DataFrame(
            [label_to_obs_fixture[label] for label in label_subset]
        ).set_index('cell_label')
        X = np.vstack(
            [label_to_row_fixture[label] for label in label_subset]
        )
        adata = anndata.AnnData(
            var=var,
            obs=obs,
            X=X
        )
        h5ad_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'
        )
        adata.write_h5ad(h5ad_path)
        h5ad_path_list.append(h5ad_path)
    return h5ad_path_list


@pytest.fixture
def wrong_var_fixture(
        tmp_dir_fixture,
        label_to_row_fixture):
    """
    An h5ad file with a different var dataframe
    from the others, to test exception when var is not uniform
    """
    rng = np.random.default_rng(8812)
    row = list(label_to_row_fixture.values())[0]
    n_genes = len(row)
    n_cells = 5
    var = pd.DataFrame(
        [{'gene_id': f'g_{10*ii}',
          'other': f'b{rng.integers(5, 9999)}'}
         for ii in range(n_genes)
         ]
    ).set_index('gene_id')

    rng = np.random.default_rng(22131)
    x = rng.random((n_cells, n_genes))
    obs = pd.DataFrame(
        [{'cell_label': f'cell_{ii+500}'}
         for ii in range(n_cells)]
    ).set_index('cell_label')
    dst_path = mkstemp_clean(
       dir=tmp_dir_fixture,
       suffix='.h5ad'
    )
    adata = anndata.AnnData(
        var=var,
        obs=obs,
        X=x
    )
    adata.write_h5ad(dst_path)
    return dst_path


@pytest.fixture
def wrong_obs_fixture(
        tmp_dir_fixture,
        label_to_row_fixture,
        src_h5ad_list_fixture):
    """
    An h5ad file with an obs dataframe that has a different
    index than the other files.
    """
    rng = np.random.default_rng(8812)
    row = list(label_to_row_fixture.values())[0]
    n_genes = len(row)
    n_cells = 5

    var = read_df_from_h5ad(
        src_h5ad_list_fixture[0],
        df_name='var'
    )

    rng = np.random.default_rng(22131)
    x = rng.random((n_cells, n_genes))

    obs = pd.DataFrame(
        [{'cell_name': f'cell_{ii+500}'}
         for ii in range(n_cells)]
    ).set_index('cell_name')

    dst_path = mkstemp_clean(
       dir=tmp_dir_fixture,
       suffix='.h5ad'
    )
    adata = anndata.AnnData(
        var=var,
        obs=obs,
        X=x
    )
    adata.write_h5ad(dst_path)
    return dst_path


@pytest.fixture
def duplicated_cell_fixture(
        tmp_dir_fixture,
        label_to_row_fixture,
        src_h5ad_list_fixture):
    """
    An h5ad file that duplicates cells in other h5ad file,
    to test error when source of truth is confused
    """
    rng = np.random.default_rng(8812)
    row = list(label_to_row_fixture.values())[0]
    n_genes = len(row)
    n_cells = 5

    var = read_df_from_h5ad(
        src_h5ad_list_fixture[0],
        df_name='var'
    )

    rng = np.random.default_rng(22131)
    x = rng.random((n_cells, n_genes))

    obs = pd.DataFrame(
        [{'cell_label': f'cell_{ii}'}
         for ii in range(300, 300+n_cells, 1)]
    ).set_index('cell_label')

    dst_path = mkstemp_clean(
       dir=tmp_dir_fixture,
       suffix='.h5ad'
    )
    adata = anndata.AnnData(
        var=var,
        obs=obs,
        X=x
    )
    adata.write_h5ad(dst_path)
    return dst_path


@pytest.mark.parametrize(
    'compression,dst_sparse',
    itertools.product(
        [True, False],
        [True, False]
    )
)
def test_amalgamate_from_label_list(
        tmp_dir_fixture,
        compression,
        dst_sparse,
        label_to_row_fixture,
        label_to_obs_fixture,
        src_h5ad_list_fixture):

    label_list = [
        'cell_300',
        'cell_5',
        'cell_124',
        'cell_89',
        'cell_222',
        'cell_111',
        'cell_0',
        'cell_9',
        'cell_306',
        'cell_101',
        'cell_102',
        'cell_103',
        'cell_104',
        'cell_105',
        'cell_106',
        'cell_107',
        'cell_108',
        'cell_109',
        'cell_110'
    ]

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='amalgamated_from_labels_',
        suffix='.h5ad'
    )

    reference_var = read_df_from_h5ad(
        src_h5ad_list_fixture[0],
        df_name='var')

    amalgamate_h5ad_from_label_list(
        src_h5ad_list=src_h5ad_list_fixture,
        row_label_list=label_list,
        dst_path=dst_path,
        dst_sparse=dst_sparse,
        compression=compression,
        tmp_dir=tmp_dir_fixture,
        rows_at_a_time=3
    )

    actual = anndata.read_h5ad(dst_path, backed='r')

    pd.testing.assert_frame_equal(
        reference_var,
        actual.var
    )

    actual_obs = {
        cell['cell_label']: cell
        for cell in actual.obs.reset_index().to_dict(orient='records')
    }

    assert len(actual_obs) == len(label_list)

    for label in label_list:
        assert actual_obs[label] == label_to_obs_fixture[label]

    actual_labels = actual.obs.index.values
    for idx, label in enumerate(actual_labels):
        expected_row = label_to_row_fixture[label]
        actual_row = actual.X[idx, :]
        if dst_sparse:
            actual_row = actual_row.toarray()[0, :]
        np.testing.assert_allclose(
            expected_row,
            actual_row,
            atol=0.0,
            rtol=1.0e-6
        )


def test_amalgamate_with_missing_cell(
        tmp_dir_fixture,
        label_to_row_fixture,
        label_to_obs_fixture,
        src_h5ad_list_fixture,
        wrong_var_fixture):
    """
    Test that amalgamation throws expected error when you
    ask for a cell that is not in your data.
    """

    label_list = [
        'cell_300',
        'cell_5',
        'cell_110',
        'garbage'
    ]

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='amalgamated_from_labels_',
        suffix='.h5ad'
    )

    msg = "Could not find data for rows"
    with pytest.raises(RuntimeError, match=msg):
        amalgamate_h5ad_from_label_list(
            src_h5ad_list=src_h5ad_list_fixture,
            row_label_list=label_list,
            dst_path=dst_path,
            dst_sparse=True,
            compression=True,
            tmp_dir=tmp_dir_fixture,
            rows_at_a_time=100
        )


def test_amalgamate_with_different_var(
        tmp_dir_fixture,
        label_to_row_fixture,
        label_to_obs_fixture,
        src_h5ad_list_fixture,
        wrong_var_fixture):
    """
    Test that amalgamation functions properly when
    trying to draw from h5ad files with different
    var dataframes
    """

    src_path_list = src_h5ad_list_fixture + [wrong_var_fixture]

    label_list = [
        'cell_300',
        'cell_5',
        'cell_110'
    ]

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='amalgamated_from_labels_',
        suffix='.h5ad'
    )

    # first check that amalgamation happens
    # when we do not actually need cells from
    # the file with the disparate var dataframe
    amalgamate_h5ad_from_label_list(
        src_h5ad_list=src_path_list,
        row_label_list=label_list,
        dst_path=dst_path,
        dst_sparse=True,
        compression=True,
        tmp_dir=tmp_dir_fixture,
        rows_at_a_time=100
    )

    # now check that an exception is thrown when we
    # actually need to bring in the h5ad file with
    # the wrong var dataframe
    msg = (
        "have different var dataframes"
    )
    with pytest.raises(RuntimeError, match=msg):
        amalgamate_h5ad_from_label_list(
            src_h5ad_list=src_path_list,
            row_label_list=label_list + ['cell_500'],
            dst_path=dst_path,
            dst_sparse=True,
            compression=True,
            tmp_dir=tmp_dir_fixture,
            rows_at_a_time=100
         )


def test_amalgamate_with_different_obs(
        tmp_dir_fixture,
        label_to_row_fixture,
        label_to_obs_fixture,
        src_h5ad_list_fixture,
        wrong_obs_fixture):
    """
    Test that amalgamation functions properly when
    trying to draw from h5ad files with different
    obs dataframes
    """

    src_path_list = src_h5ad_list_fixture + [wrong_obs_fixture]

    label_list = [
        'cell_300',
        'cell_5',
        'cell_110'
    ]

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='amalgamated_from_labels_',
        suffix='.h5ad'
    )

    # first check that amalgamation happens
    # when we do not actually need cells from
    # the file with the disparate obs dataframe
    amalgamate_h5ad_from_label_list(
        src_h5ad_list=src_path_list,
        row_label_list=label_list,
        dst_path=dst_path,
        dst_sparse=True,
        compression=True,
        tmp_dir=tmp_dir_fixture,
        rows_at_a_time=100
    )

    # now check that an exception is thrown when we
    # actually need to bring in the h5ad file with
    # the wrong obs dataframe
    msg = (
        "Mismatch in obs indexes"
    )
    with pytest.raises(RuntimeError, match=msg):
        amalgamate_h5ad_from_label_list(
            src_h5ad_list=src_path_list,
            row_label_list=label_list + ['cell_500'],
            dst_path=dst_path,
            dst_sparse=True,
            compression=True,
            tmp_dir=tmp_dir_fixture,
            rows_at_a_time=100
         )


def test_amalgamate_with_duplicated_cell(
        tmp_dir_fixture,
        label_to_row_fixture,
        label_to_obs_fixture,
        src_h5ad_list_fixture,
        duplicated_cell_fixture):
    """
    Test that amalgamation functions properly when
    a desired cell is listed in more than one h5ad file
    """

    src_path_list = src_h5ad_list_fixture + [duplicated_cell_fixture]

    label_list = [
        'cell_200',
        'cell_5',
        'cell_110'
    ]

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='amalgamated_from_labels_',
        suffix='.h5ad'
    )

    # first check that amalgamation happens
    # when we do not actually need cells that
    # are duplicated
    amalgamate_h5ad_from_label_list(
        src_h5ad_list=src_path_list,
        row_label_list=label_list,
        dst_path=dst_path,
        dst_sparse=True,
        compression=True,
        tmp_dir=tmp_dir_fixture,
        rows_at_a_time=100
    )

    # now check that an exception is thrown when we
    # actually need to bring in the duplicated cells
    msg = (
        "Two sources of truth"
    )
    with pytest.raises(RuntimeError, match=msg):
        amalgamate_h5ad_from_label_list(
            src_h5ad_list=src_path_list,
            row_label_list=label_list + ['cell_301'],
            dst_path=dst_path,
            dst_sparse=True,
            compression=True,
            tmp_dir=tmp_dir_fixture,
            rows_at_a_time=100
         )
