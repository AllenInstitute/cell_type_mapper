import pytest

import anndata
import h5py
import json
import numpy as np
import pandas as pd
import pathlib

from hierarchical_mapping.utils.utils import (
   _clean_up,
   mkstemp_clean)

from hierarchical_mapping.corr.correlate_cells import (
    correlate_cells)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('corr_markers'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def query_gene_names():
    return ['m0', 'q0', 'm2', 'q3', 'm1', 'q1', 'm3']


@pytest.fixture
def reference_gene_names():
    return ['m0', 'm1', 'm2', 'r0', 'm3', 'm4', 'r1']


@pytest.fixture
def marker_gene_names():
    return ['m1', 'm2', 'm0', 'm3', 'm4']


@pytest.fixture
def cluster_to_profile(reference_gene_names):
    result = dict()
    rng = np.random.default_rng(87123)
    n_genes = len(reference_gene_names)
    for n in ('c0', 'c1', 'c2', 'c3', 'c4', 'c5'):
        result[n] = rng.random(n_genes)
    return result

@pytest.fixture
def precompute_fixture(
        tmp_dir_fixture,
        reference_gene_names,
        cluster_to_profile):

    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    cluster_list = list(cluster_to_profile.keys())
    cluster_list.sort()

    rng = np.random.default_rng(61237)
    n_cells = rng.integers(40, 116, len(cluster_list))
    sum_arr = np.zeros((len(cluster_list), len(reference_gene_names)),
                       dtype=float)
    for ii in range(len(cluster_list)):
        sum_arr[ii, :] = n_cells[ii]*cluster_to_profile[cluster_list[ii]]

    with h5py.File(h5_path, 'w') as out_file:
        out_file.create_dataset(
            'col_names',
            data=json.dumps(reference_gene_names).encode('utf-8'))
        out_file.create_dataset(
            'cluster_to_row',
            data=json.dumps(
                {n:int(ii)
                 for ii, n in
                 enumerate(cluster_list)}).encode('utf-8'))
        out_file.create_dataset(
           'n_cells', data=n_cells)
        out_file.create_dataset(
            'sum', data=sum_arr)

    return h5_path


@pytest.fixture
def query_x_fixture(
        query_gene_names):
    rng = np.random.default_rng(645221)
    n_genes = len(query_gene_names)
    n_cells = 13
    return rng.random((n_cells, n_genes))


@pytest.fixture
def query_h5ad_fixture(
        query_gene_names,
        query_x_fixture,
        tmp_dir_fixture):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_',
        suffix='.h5ad')


    var_data = [
        {'gene_name': n, 'junk': ii}
        for ii, n in enumerate(query_gene_names)]
    var = pd.DataFrame(var_data).set_index('gene_name')

    obs_data = [
        {'cell_id': ii, 'garbage': ii**2}
        for ii in range(query_x_fixture.shape[0])]
    obs = pd.DataFrame(obs_data).set_index('cell_id')

    a = anndata.AnnData(
            X=query_x_fixture,
            obs=obs,
            var=var)
    a.write_h5ad(h5ad_path)
    return h5ad_path


def test_correlate_cells_with_markers(
        tmp_dir_fixture,
        query_h5ad_fixture,
        query_x_fixture,
        precompute_fixture,
        cluster_to_profile,
        marker_gene_names):

    cluster_list = list(cluster_to_profile.keys())
    cluster_list.sort()

    output_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='output_',
            suffix='.h5')

    correlate_cells(
        query_path=query_h5ad_fixture,
        precomputed_path=precompute_fixture,
        output_path=output_path,
        rows_at_a_time=17,
        n_processors=3,
        marker_gene_list=marker_gene_names)

    expected_corr = np.zeros((query_x_fixture.shape[0], len(cluster_list)),
                             dtype=float)

    reference_idx = np.array([0, 1, 2, 4])
    query_idx = np.array([0, 4, 2, 6])

    for i_query in range(query_x_fixture.shape[0]):
        query_vec = query_x_fixture[i_query, query_idx]
        query_std = np.std(query_vec, ddof=0)
        query_mu = np.mean(query_vec)
        for i_cluster, cluster in enumerate(cluster_list):
            cluster_vec = cluster_to_profile[cluster][reference_idx]
            cluster_std = np.std(cluster_vec, ddof=0)
            cluster_mu = np.mean(cluster_vec)
            corr = np.mean((cluster_vec-cluster_mu)*(query_vec-query_mu))
            corr = corr / (query_std*cluster_std)
            expected_corr[i_query, i_cluster] = corr

    with h5py.File(output_path, 'r') as in_file:
        actual = in_file['correlation'][()]
    np.testing.assert_allclose(
        actual,
        expected_corr,
        atol=0.0,
        rtol=1.0e-5)
            
    
