"""
In this file, we run the test where we feed mean gene expressions back
in as the query data to make sure we get back the exact cluster assignments.
"""

import pytest

import anndata
import h5py
import json
import numpy as np
import pandas as pd
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.type_assignment.election_runner import (
    run_type_assignment_on_h5ad)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp("tautology"))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_genes():
    return 190


@pytest.fixture
def gene_names(n_genes):
    return [f'gene_{ii}' for ii in range(n_genes)]


@pytest.fixture
def taxonomy_dict_fixture():
    taxonomy = dict()
    taxonomy['hierarchy'] = ['level_1', 'level_2', 'cluster']
    taxonomy['level_1'] = {
        'A': ['aa', 'bb'],
        'B': ['cc'],
        'C': ['dd', 'ee']}
    taxonomy['level_2'] = {
        'aa': ['c1', 'c2'],
        'bb': ['c3', 'c4', 'c5'],
        'cc': ['c6', 'c7', 'c8', 'c9'],
        'dd': ['c10', 'c11'],
        'ee': ['c12', 'c13']
    }
    taxonomy['cluster'] = {
        f'c{ii}': [ii, ii+22]
        for ii in range(1,14,1)
    }
    return taxonomy


@pytest.fixture
def mean_profile_fixture(
       n_genes,
       taxonomy_dict_fixture):
    result = dict()
    tt = np.linspace(0, 1, n_genes)
    for ii, cl in enumerate(taxonomy_dict_fixture['cluster'].keys()):
        profile = np.sin(2.0*np.pi*tt*(ii+1)/15.0)
        result[cl] = profile
    return result


@pytest.fixture
def precomputed_fixture(
        tmp_dir_fixture,
        mean_profile_fixture,
        gene_names,
        n_genes):
    n_cells = len(mean_profile_fixture)
    data = np.zeros((n_cells, n_genes), dtype=float)
    cluster_to_row = dict()
    for ii, cl in enumerate(mean_profile_fixture):
        data[ii, :] = mean_profile_fixture[cl]
        cluster_to_row[cl] = int(ii)

    out_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precompute_',
        suffix='.h5')
    with h5py.File(out_path, 'w') as out_file:
        out_file.create_dataset(
            'cluster_to_row',
            data=json.dumps(cluster_to_row).encode('utf-8')),
        out_file.create_dataset(
            'col_names',
            data=json.dumps(gene_names).encode('utf-8'))
        out_file.create_dataset(
            'n_cells',
            data=np.ones(n_cells, dtype=int)),
        out_file.create_dataset(
            'sum',
            data=data)
        for k in ('gt1', 'gt0', 'ge1', 'sumsq'):
            out_file.create_dataset(
                k,
                data=np.zeros((n_cells, n_genes), dtype=int))
    return out_path


@pytest.fixture
def query_fixture(
        precomputed_fixture,
        gene_names,
        tmp_dir_fixture):
    with h5py.File(precomputed_fixture, 'r') as f:
        x_matrix = f['sum'][()]
        cluster_to_row = json.loads(f['cluster_to_row'][()].decode('utf-8'))
    var_data = [{'gene_name': g} for g in gene_names]
    obs_data = [None]*len(cluster_to_row)
    for cl in cluster_to_row:
        obs_data[cluster_to_row[cl]] = {'cl': cl, 'cell_id': cl}
    obs_df = pd.DataFrame(obs_data)
    obs_df = obs_df.set_index('cell_id')
    var_df = pd.DataFrame(var_data)
    var_df = var_df.set_index('gene_name')

    anndata_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_',
        suffix='.h5ad')

    a = anndata.AnnData(
        X=x_matrix,
        obs=obs_df,
        var=var_df,
        dtype=x_matrix.dtype)
    a.write_h5ad(anndata_path)
    return anndata_path


@pytest.fixture
def marker_fixture(
        n_genes,
        gene_names,
        taxonomy_dict_fixture,
        tmp_dir_fixture):
    marker_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_',
        suffix='.h5')

    parents = ['None']
    for level in taxonomy_dict_fixture['hierarchy'][:-1]:
        for node in taxonomy_dict_fixture[level].keys():
            parents.append(f'{level}/{node}')

    all_markers = set()
    rng = np.random.default_rng(8712312)
    with h5py.File(marker_path, 'w') as out_file:
        for grp in parents:
            chosen = rng.choice(np.arange(n_genes, dtype=int),
                                rng.integers(8, 11),
                                replace=False)
            out_file.create_dataset(f"{grp}/reference", data=chosen)
            out_file.create_dataset(f"{grp}/query", data=chosen)
            all_markers = all_markers.union(set(chosen))
        all_markers = np.sort(np.array(list(all_markers)))
        out_file.create_dataset(
            "all_query_markers",
            data=all_markers)
        out_file.create_dataset(
            "all_reference_markers",
            data=all_markers)
        name_data = json.dumps(gene_names).encode('utf-8')
        out_file.create_dataset(
            "query_gene_names", data=name_data)
        out_file.create_dataset(
            "reference_gene_names", data=name_data)

    return marker_path


def test_tautological_assignment(
        marker_fixture,
        query_fixture,
        precomputed_fixture,
        taxonomy_dict_fixture):
    """
    Test that if the query data *is* the mean gene expression profiles,
    we get the exact assignments out that we expected
    """

    taxonomy_tree = TaxonomyTree(data=taxonomy_dict_fixture)

    factor = 0.9
    bootstrap_factor_lookup = {
        level: factor
        for level in taxonomy_tree.hierarchy}
    bootstrap_factor_lookup['None'] = factor

    result = run_type_assignment_on_h5ad(
        query_h5ad_path=query_fixture,
        precomputed_stats_path=precomputed_fixture,
        marker_gene_cache_path=marker_fixture,
        taxonomy_tree=taxonomy_tree,
        n_processors=3,
        chunk_size=50,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=100,
        rng=np.random.default_rng(887123))
    assert len(result) == len(taxonomy_dict_fixture['cluster'])
    for el in result:
        assert el['cluster']['assignment'] == el['cell_id']
        cl = el['cluster']['assignment']
        l1 = el['level_1']['assignment']
        l2 = el['level_2']['assignment']
        assert cl in taxonomy_dict_fixture['level_2'][l2]
        assert l2 in taxonomy_dict_fixture['level_1'][l1]
        for k in ('cluster', 'level_1', 'level_2'):
            np.testing.assert_allclose(
                el[k]['bootstrapping_probability'],
                1.0,
                atol=0.0,
                rtol=1.0e-6)
