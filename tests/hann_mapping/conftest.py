import pytest

import anndata
import h5py
import json
import numpy as np
import pandas as pd
import warnings

import cell_type_mapper.utils.utils as ctm_utils
import cell_type_mapper.cell_by_gene.cell_by_gene as cbg_module
import cell_type_mapper.taxonomy.taxonomy_tree as tree_module


@pytest.fixture
def tree_fixture():
    tree_data = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {'A': ['a', 'b'], 'B': ['c']},
        'subclass': {'a': ['a1'],
                     'b': ['b1', 'b2'],
                     'c': ['c1', 'c2']},
        'cluster': {
            'a1': [],
            'b1': [],
            'b2': [],
            'c1': [],
            'c2': []}
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        taxonomy_tree = tree_module.TaxonomyTree(data=tree_data)
    return taxonomy_tree


@pytest.fixture
def cell_by_gene_fixture(tree_fixture):

    n_cells = 20
    n_clusters = len(tree_fixture.nodes_at_level(tree_fixture.leaf_level))
    n_genes = 30
    gene_identifiers = [f'g{ii}' for ii in range(n_genes)]

    rng = np.random.default_rng(881231)

    reference_data = np.zeros((n_clusters, n_genes), dtype=float)
    reference_data[0, 10:25] = np.sin(np.arange(15)*2.0*np.pi/7.0)
    reference_data[1, 0:18] = np.cos(np.arange(18)*2.0*np.pi/12.0)
    reference_data[2, 17:n_genes] = 2.0*(rng.random(13)-0.5)
    reference_data[3, 10:] = 1.2
    reference_data[4, :15] = 0.7
    reference_data[4, 15:] = np.linspace(0, 1.5, 15)

    reference = cbg_module.CellByGeneMatrix(
        data=reference_data,
        gene_identifiers=gene_identifiers,
        cell_identifiers=tree_fixture.nodes_at_level(
            tree_fixture.leaf_level),
        normalization='log2CPM'
    )

    query_data = np.zeros((n_cells, n_genes), dtype=float)
    for i_cells in range(n_cells):
        pair = rng.choice(np.arange(n_clusters), 2, replace=False)
        weights = rng.random(2)
        this = (weights[0]*reference_data[pair[0], :]
                + weights[1]*reference_data[pair[1], :]) / weights.sum()
        query_data[i_cells, :] = this

    query = cbg_module.CellByGeneMatrix(
        data=query_data,
        gene_identifiers=gene_identifiers,
        normalization='log2CPM',
        cell_identifiers=[f'c{ii}' for ii in range(n_cells)]
    )

    return {'reference': reference, 'query': query}


@pytest.fixture
def precomputed_stats_fixture(
        cell_by_gene_fixture,
        tree_fixture,
        tmp_dir_fixture):

    src = cell_by_gene_fixture['reference']

    dst_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hann_election_precomputed_stats_',
        suffix='.h5'
    )
    with h5py.File(dst_path, 'w') as dst:
        dst.create_dataset(
            'n_cells',
            data=np.ones(src.n_cells)
        )
        dst.create_dataset(
            'cluster_to_row',
            data=json.dumps(
                {nn: ii
                 for ii, nn in enumerate(src.cell_identifiers)
                 }
            ).encode('utf-8')
        )
        dst.create_dataset(
            'col_names',
            data=json.dumps(src.gene_identifiers).encode('utf-8')
        )
        dst.create_dataset(
            'sum',
            data=src.data
        )
        dst.create_dataset(
            'taxonomy_tree',
            data=tree_fixture.to_str().encode('utf-8')
        )
    return dst_path


@pytest.fixture
def query_h5ad_fixture(
        cell_by_gene_fixture,
        tmp_dir_fixture):

    dst_path = ctm_utils.mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='hann_election_query_',
        suffix='.h5ad'
    )

    rng = np.random.default_rng(881231)
    src = cell_by_gene_fixture['reference']
    n_cells = 200
    var = pd.DataFrame(
        [{'gene': g} for g in src.gene_identifiers]
    ).set_index('gene')
    obs = pd.DataFrame(
        [{'cell': f'c{ii}'} for ii in range(n_cells)]
    ).set_index('cell')
    data = np.zeros((n_cells, len(var)), dtype=float)
    for i_cell in range(n_cells):
        chosen = rng.choice(np.arange(src.n_cells), 3, replace=False)
        wgt = rng.random(3)
        data[i_cell, :] = (
            wgt[0]*src.data[chosen[0], :]
            + wgt[1]*src.data[chosen[1], :]
            + wgt[2]*src.data[chosen[2], :]
        )/wgt.sum()
    adata = anndata.AnnData(
        var=var,
        obs=obs,
        X=data
    )
    adata.write_h5ad(dst_path)
    return dst_path
