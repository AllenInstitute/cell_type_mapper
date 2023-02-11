import numpy as np
import scipy.sparse as scipy_sparse
import anndata
import os
import tempfile
import zarr

from hierarchical_mapping.utils.sparse_utils import(
    load_csr)


def test_load_csr():
    tmp_path = tempfile.mkdtemp(suffix='.zarr')

    rng = np.random.default_rng(88123)

    data = np.zeros(60000, dtype=int)
    chosen_dex = rng.choice(np.arange(len(data)),
                            len(data)//4,
                            replace=False)

    data[chosen_dex] = rng.integers(2, 1000, len(chosen_dex))
    data = data.reshape((200, 300))

    csr = scipy_sparse.csr_matrix(data)
    ann = anndata.AnnData(csr)
    ann.write_zarr(tmp_path)

    with zarr.open(tmp_path, 'r') as written_zarr:
        for r0 in range(0, 150, 47):
            r1 = min(200, r0+47)
            subset = load_csr(
                        row_spec=(r0, r1),
                        n_cols=data.shape[1],
                        data=written_zarr.X.data,
                        indices=written_zarr.X.indices,
                        indptr=written_zarr.X.indptr)
            np.testing.assert_array_equal(
                subset,
                data[r0:r1, :])

        for ii in range(10):
            r0 = rng.integers(3, 50)
            r1 = min(data.shape[0], r0+rng.integers(17, 81))

            subset = load_csr(
                row_spec=(r0, r1),
                n_cols=data.shape[1],
                data=written_zarr.X.data,
                indices=written_zarr.X.indices,
                indptr=written_zarr.X.indptr)

            np.testing.assert_array_equal(
                subset,
                data[r0:r1, :])
