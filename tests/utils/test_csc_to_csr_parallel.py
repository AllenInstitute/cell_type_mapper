import pytest

import itertools
import numpy as np
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.csc_to_csr_parallel import (
    _grab_indices_from_chunk)


@pytest.mark.parametrize(
        'indptr_0,indices_minmax,clip_indices',
        itertools.product([0, 5], [(0, 89), (11,35)], [True, False]))
def test_grab_indices_from_chunk(
        indptr_0,
        indices_minmax,
        clip_indices):

    n_indices = 100
    rng = np.random.default_rng(445513)
    n_rows = 67
    n_cols = 89
    n_tot = n_rows*n_cols
    data = np.zeros(n_tot, dtype=int)
    chosen = rng.choice(np.arange(n_tot), n_tot//3)
    data[chosen] = 1
    data = data.reshape((n_rows, n_cols))
    data[:, 55] = 0
    csr = scipy_sparse.csr_matrix(data)

    expected = dict()
    if clip_indices:
        n_indptr = 11
        indices_position = (csr.indptr[indptr_0], csr.indptr[indptr_0+n_indptr])
        indptr = csr.indptr[indptr_0:indptr_0+n_indptr+1]
        indices = csr.indices[indices_position[0]:indices_position[1]]
        for i_indptr in range(n_indptr):
            i_row = i_indptr+indptr_0
            for i_col in csr.indices[indptr[i_indptr]:indptr[i_indptr+1]]:
                if i_col not in expected:
                    expected[i_col] = []
                if i_col < indices_minmax[0] or i_col >= indices_minmax[1]:
                    continue
                expected[i_col].append(i_row)
        indptr = csr.indptr[indptr_0:]
    else:
        indptr = csr.indptr
        indices = csr.indices
        for i_col in range(n_cols):
            expected[i_col] = []
            if i_col < indices_minmax[0] or i_col >= indices_minmax[1]:
                continue
            for i_row in range(n_rows):
                if data[i_row, i_col] != 0:
                    expected[i_col].append(i_row+indptr_0)

        indices_position = (0, csr.indices.shape[0])

    for i_col in expected:
        expected[i_col] = np.array(expected[i_col])

    output_dict = dict()
    _grab_indices_from_chunk(
        indptr=indptr,
        indptr_0=indptr_0,
        indices_chunk=indices,
        indices_minmax=indices_minmax,
        indices_position=indices_position,
        output_dict=output_dict)

    for k in output_dict:
        output_dict[k] = np.concatenate(output_dict[k])

    print(output_dict.keys())

    for i_col in output_dict:
        assert i_col in expected

    for i_col in expected:
        if data[:, i_col].sum() == 0 or len(expected[i_col]) == 0:
            assert i_col not in output_dict
        else:
            np.testing.assert_array_equal(
                output_dict[i_col],
                expected[i_col])
