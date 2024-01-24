import pytest

import h5py
import itertools
import numpy as np
import scipy.sparse as scipy_sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.csc_to_csr_parallel import (
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
        "indices_slice,use_data",
        itertools.product(
            [(0, 25), (0, 89), (23, 89), (34, 65)],
            [True, False]
        ))
def test_transpose_chunk_of_indices(
        sparse_array_fixture,
        tmp_dir_fixture,
        indices_slice,
        use_data):

    csr = scipy_sparse.csr_matrix(sparse_array_fixture)

    indices_max = csr.shape[1]

    csr_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5',
        prefix='csr_')

    with h5py.File(csr_path, 'w') as dst:
        dst.create_dataset('data', data=csr.data)
        dst.create_dataset('indices', data=csr.indices)
        dst.create_dataset('indptr', data=csr.indptr)

    subset = sparse_array_fixture[:, indices_slice[0]:indices_slice[1]]
    csc = scipy_sparse.csc_matrix(subset)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5',
        prefix='transposed_chunk_')

    if use_data:
        data_tag = 'data'
    else:
        data_tag = None

    _transpose_subset_of_indices(
        h5_path=csr_path,
        indices_tag='indices',
        indptr_tag='indptr',
        data_tag=data_tag,
        indices_max=indices_max,
        indices_slice=indices_slice,
        output_path=output_path)

    with h5py.File(output_path, 'r') as src:
        minmax = src['indices_slice'][()]
        assert minmax[0] == indices_slice[0]
        assert minmax[1] == indices_slice[1]
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
