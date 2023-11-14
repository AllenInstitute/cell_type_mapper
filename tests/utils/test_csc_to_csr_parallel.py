import pytest

import h5py
import itertools
import numpy as np
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.csc_to_csr_parallel import (
    _grab_indices_from_chunk,
    _transpose_subset_of_indices)


@pytest.fixture
def sparse_array_fixture():
    rng = np.random.default_rng(445513)
    n_rows = 67
    n_cols = 89
    n_tot = n_rows*n_cols
    data = np.zeros(n_tot, dtype=int)
    chosen = rng.choice(np.arange(n_tot), n_tot//3)
    data[chosen] = rng.integers(10, 5000, len(chosen))
    data = data.reshape((n_rows, n_cols))
    data[:, 55] = 0
    return data


@pytest.mark.parametrize(
        'indptr_0,indices_minmax,clip_indices,use_data',
        itertools.product([0, 5],
                          [(0, 89), (11,35)],
                          [True, False],
                          [True, False]))
def test_grab_indices_from_chunk(
        sparse_array_fixture,
        indptr_0,
        indices_minmax,
        clip_indices,
        use_data):

    data = sparse_array_fixture
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    csr = scipy_sparse.csr_matrix(data)

    expected_indices = dict()
    expected_data = dict()
    if clip_indices:
        n_indptr = 11
    else:
        n_indptr = n_rows-indptr_0

    indices_position = (csr.indptr[indptr_0], csr.indptr[indptr_0+n_indptr])
    indptr = csr.indptr[indptr_0:indptr_0+n_indptr+1]
    indices = csr.indices[indices_position[0]:indices_position[1]]
    for i_indptr in range(n_indptr):
        i_row = i_indptr+indptr_0
        for i_col in csr.indices[indptr[i_indptr]:indptr[i_indptr+1]]:
            if i_col not in expected_indices:
                expected_indices[i_col] = []
            if i_col < indices_minmax[0] or i_col >= indices_minmax[1]:
                continue
            expected_indices[i_col].append(i_row)

    for i_col in expected_indices:
        expected_indices[i_col] = np.array(expected_indices[i_col])

    if use_data:
        data_chunk = csr.data[indices_position[0]:indices_position[1]]
        for i_col in expected_indices:
            expected_data[i_col] = []
            for i_row in expected_indices[i_col]:
                expected_data[i_col].append(data[i_row, i_col])
            expected_data[i_col] = np.array(expected_data[i_col])
    else:
        data_chunk = None

    indptr = csr.indptr[indptr_0:]


    output_dict = dict()
    _grab_indices_from_chunk(
        indptr=indptr,
        indptr_0=indptr_0,
        indices_chunk=indices,
        data_chunk=data_chunk,
        indices_minmax=indices_minmax,
        indices_position=indices_position,
        output_dict=output_dict)

    for i_col in output_dict:
        assert i_col in expected_indices
        if use_data is None:
            assert output_dict[i_col]['data'] == []

    for i_col in expected_indices:
        if data[:, i_col].sum() == 0 or len(expected_indices[i_col]) == 0:
            assert i_col not in output_dict
        else:
            np.testing.assert_array_equal(
                np.concatenate(output_dict[i_col]['indices']),
                expected_indices[i_col])
            if use_data:
                np.testing.assert_array_equal(
                    np.concatenate(output_dict[i_col]['data']),
                    expected_data[i_col])


@pytest.mark.parametrize(
        "indices_minmax,use_data",
        itertools.product(
            [(0, 25), (0, 89), (23, 89), (34, 65)],
            [True, False]
        ))
def test_transpose_chunk_of_indices(
        sparse_array_fixture,
        tmp_dir_fixture,
        indices_minmax,
        use_data):

    csr = scipy_sparse.csr_matrix(sparse_array_fixture)

    subset = sparse_array_fixture[:, indices_minmax[0]:indices_minmax[1]]
    csc = scipy_sparse.csc_matrix(subset)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5',
        prefix='transposed_chunk_')

    if use_data:
        data_handle = csr.data
    else:
        data_handle = None

    _transpose_subset_of_indices(
        indices_handle=csr.indices,
        indptr_handle=csr.indptr,
        data_handle=data_handle,
        indices_minmax=indices_minmax,
        chunk_size=100,
        output_path=output_path)

    with h5py.File(output_path, 'r') as src:
        minmax = src['indices_minmax'][()]
        assert minmax[0] == indices_minmax[0]
        assert minmax[1] == indices_minmax[1]
        np.testing.assert_array_equal(
            src['indices'][()],
            csc.indices)
        np.testing.assert_array_equal(
            src['indptr'][()],
            csc.indptr)
        if not use_data:
            assert 'data' not in src.keys()
        else:
            np.testing.assert_array_equal(
                src['data'][()],
                csc.data)
