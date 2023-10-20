"""
Implement some smoketest-level tests of the from_specified_markers
CLI tool.
"""
import pytest

import anndata
import copy
import h5py
import json
import numpy as np
import pandas as pd
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.data.mouse_gene_id_lookup import (
    mouse_gene_id_lookup)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('cli_smoke_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def taxonomy_tree_fixture():
    data = {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {
            'classA': ['subclassB', 'subclassC'],
            'classB': ['subclassA', 'subclassD']
        },
        'subclass': {
            'subclassA': ['c0', 'c2'],
            'subclassB': ['c1', 'c3'],
            'subclassC': ['c4', 'c6'],
            'subclassD': ['c5', 'c7']
        },
        'cluster': {
            f'c{ii}': [] for ii in range(8)
        }
    }
    return TaxonomyTree(data=data)


@pytest.fixture(scope='module')
def gene_name_fixture():
    rng = np.random.default_rng(2213)
    result = [k for k in mouse_gene_id_lookup.keys() if 'NCBI' not in k]
    result = rng.choice(result, 432, replace=False)
    return list(result)


@pytest.fixture(scope='module')
def gene_id_fixture(gene_name_fixture):
    return [mouse_gene_id_lookup[g] for g in gene_name_fixture]


@pytest.fixture(scope='module')
def marker_lookup_fixture(
        taxonomy_tree_fixture,
        gene_id_fixture,
        tmp_dir_fixture):

    json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_lookup_',
        suffix='.json')

    rng = np.random.default_rng(42312)
    parent_list = taxonomy_tree_fixture.all_parents
    lookup = dict()
    for parent in parent_list:
        if parent is None:
            parent_k = 'None'
        else:
            parent_k = f'{parent[0]}/{parent[1]}'
        lookup[parent_k] = list(rng.choice(gene_id_fixture, 15, replace=False))
    with open(json_path, 'w') as dst:
        dst.write(json.dumps(lookup, indent=2))
    return json_path


@pytest.fixture(scope='module')
def precomputed_stats_fixture(
        tmp_dir_fixture,
        gene_id_fixture,
        taxonomy_tree_fixture):

    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    rng = np.random.default_rng(76123)
    n_clusters = 8
    n_genes = len(gene_id_fixture)
    sum_values = rng.random((n_clusters, n_genes))
    with h5py.File(h5_path, 'w') as src:
        src.create_dataset('sum', data=sum_values)
        src.create_dataset('n_cells', data=rng.integers(10, 25, n_clusters))
        src.create_dataset(
            'taxonomy_tree',
            data=taxonomy_tree_fixture.to_str().encode('utf-8'))
        src.create_dataset('col_names',
            data=json.dumps(gene_id_fixture).encode('utf-8'))
        src.create_dataset('cluster_to_row',
            data=json.dumps(
                {f'c{ii}': ii for ii in range(n_clusters)}).encode('utf-8'))

    return h5_path


@pytest.fixture(scope='module')
def query_h5ad_fixture(
        gene_name_fixture,
        tmp_dir_fixture):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_data_',
        suffix='.h5ad')

    n_extra = 19

    n_cells = 321
    n_genes = len(gene_name_fixture) + n_extra

    rng = np.random.default_rng(77123)
    X = rng.random((n_cells, n_genes), dtype=np.float32)

    obs = pd.DataFrame(
        [{'cell_id': f'cell_{ii}'}
         for ii in range(n_cells)]).set_index('cell_id')

    these_gene_names = copy.deepcopy(gene_name_fixture)
    for ii in range(n_extra):
        these_gene_names.append(f'extra_gene_{ii}')
    rng.shuffle(these_gene_names)

    var = pd.DataFrame(
        [{'gene_name': g} for g in these_gene_names]).set_index('gene_name')

    a_data = anndata.AnnData(
        X=X,
        obs=obs,
        var=var)

    a_data.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.mark.parametrize('map_to_ensembl', [True, False])
def test_ensembl_mapping_in_cli(
        taxonomy_tree_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        map_to_ensembl):
    """
    Test for expected behavior (error/no error) when we just
    ask the from_specified_markers CLI to map gene names to
    ENSEMBLID
    """
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='outptut_',
        suffix='.json')

    config = {
        'precomputed_stats': {
            'path': str(precomputed_stats_fixture)
        },
        'query_markers': {
            'serialized_lookup': str(marker_lookup_fixture)
        },
        'query_path': str(query_h5ad_fixture),
        'extended_result_path': str(output_path),
        'map_to_ensembl': map_to_ensembl,
        'type_assignment': {
            'normalization': 'log2CPM',
            'bootstrap_iteration': 10,
            'bootstrap_factor': 0.9,
            'n_runners_up': 2,
            'rng_seed': 5513,
            'chunk_size': 50,
            'n_processors': 3
        }
    }

    runner = FromSpecifiedMarkersRunner(
        args=[],
        input_data=config)

    if map_to_ensembl:
        runner.run()
        actual = json.load(open(output_path, 'rb'))
        assert 'RAN SUCCESSFULLY' in actual['log'][-2]
    else:
        with pytest.raises(RuntimeError, match="'None' has no valid markers"):
            runner.run()
