import pathlib
import h5py
import json
import numpy as np
import scipy.sparse as scipy_sparse
import zarr

from hierarchical_mapping.diff_exp.precompute import (
    precompute_summary_stats)


def _clean_up(target_path):
    target_path = pathlib.Path(target_path)
    if target_path.is_file():
        target_path.unlink()
    elif target_path.is_dir():
        for sub_path in target_path.iterdir():
            _clean_up(sub_path)
        target_path.rmdir()


def test_precompute_smoketest(
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('precompute_smoke'))
    input_dir = tmp_dir / 'input'
    input_dir.mkdir()
    output_dir = tmp_dir / 'output'
    output_dir.mkdir()

    rng = np.random.default_rng(712321)
    nrows = 300
    ncols = 427
    data = np.zeros(nrows*ncols, dtype=float)
    chosen_dex = rng.choice(
                      np.arange(nrows*ncols),
                      nrows*ncols//11,
                      replace=False)
    data[chosen_dex] = rng.random(len(chosen_dex))
    data = data.reshape(nrows, ncols)

    baseline_csr = scipy_sparse.csr_array(data)

    zarr_path = input_dir / 'as_zarr.zarr'
    with zarr.open(zarr_path, 'w') as out_file:
        out_file['data'] = baseline_csr.data
        out_file['indices'] = baseline_csr.indices
        out_file['indptr'] = baseline_csr.indptr

    n_clusters = 20
    cluster_to_rows = dict()
    for ii in range(n_clusters):
        cluster_to_rows[f'cluster_{ii}'] = []
    for idx in range(nrows):
        c = rng.integers(0, n_clusters)
        cluster_to_rows[f'cluster_{c}'].append(idx)

    output_path = output_dir / 'stats_cache.h5'

    precompute_summary_stats(
        data_path=zarr_path,
        cluster_to_input_row=cluster_to_rows,
        n_genes=ncols,
        output_path=output_path,
        n_processors=3,
        rows_at_a_time=60)

    with h5py.File(output_path, 'r') as in_file:
        cluster_to_output = json.loads(
                    in_file['cluster_to_row'][()].decode('utf-8'))

        assert in_file['n_cells'][()].max() > 0
        for cluster in cluster_to_output:
            these_rows = np.array(cluster_to_rows[cluster])
            expected = len(these_rows)
            this_idx = cluster_to_output[cluster]
            assert in_file['n_cells'][this_idx] == expected

            np.testing.assert_allclose(
                    in_file['sum'][this_idx, :],
                    data[these_rows, :].sum(axis=0))

            np.testing.assert_allclose(
                    in_file['sumsq'][this_idx, :],
                    (data[these_rows, :]**2).sum(axis=0))

            np.testing.assert_allclose(
                    in_file['gt0'][this_idx, :],
                    (data[these_rows, :]>0).sum(axis=0))

            np.testing.assert_allclose(
                    in_file['gt1'][this_idx, :],
                    (data[these_rows, :]>1).sum(axis=0))


    _clean_up(tmp_dir)
