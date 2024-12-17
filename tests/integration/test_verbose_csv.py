"""
Tests in this file will verify that the expected set of columns propagate
through to the CSV output file, even in the edge case where key words
like 'label', 'alias', and 'name' occur in the taxonomic levels.
"""

import pytest

import anndata
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import warnings

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean
)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree
)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad
)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner
)

from cell_type_mapper.cli.map_to_on_the_fly_markers import (
    OnTheFlyMapper
)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    result = tmp_path_factory.mktemp('verbose_csv_dir_')
    yield result
    _clean_up(result)


@pytest.fixture(scope='module')
def taxonomy_tree_data_fixture():

    data = {
        'hierarchy': ['class_name_label_alias',
                      'subclass_name_label_alias',
                      'cluster_name_label_alias'],
        'class_name_label_alias': {
            'c0': ['s0', 's1'],
            'c1': ['s2']
        },
        'subclass_name_label_alias': {
            's0': ['cl0'],
            's1': ['cl1', 'cl2'],
            's2': ['cl3']
        },
        'cluster_name_label_alias': {
            'cl0': [],
            'cl1': [],
            'cl2': [],
            'cl3': []
        },
        'name_mapper': {
            'class_name_label_alias': {
                'c0': {'name': 'zero'},
                'c1': {'name': 'one'}
            },
            'subclass_name_label_alias': {
                's0': {'name': 'subclass_zero'},
                's1': {'name': 'subclass_one'},
                's2': {'name': 'subclass_two'}
            },
            'cluster_name_label_alias': {
                'cl0': {'name': 'cluster_zero', 'alias': '00'},
                'cl1': {'name': 'cluster_one', 'alias': '01'},
                'cl2': {'name': 'cluster_two', 'alias': '02'},
                'cl3': {'name': 'cluster_three', 'alias': '03'}
            }
        }
    }

    return data


@pytest.fixture(scope='module')
def n_genes_fixture():
    return 40


@pytest.fixture(scope='module')
def reference_h5ad_fixture(
        tmp_dir_fixture,
        taxonomy_tree_data_fixture,
        n_genes_fixture):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        taxonomy_tree = TaxonomyTree(data=taxonomy_tree_data_fixture)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_data_',
        suffix='.h5ad'
    )

    n_cells = 500
    n_genes = n_genes_fixture

    rng = np.random.default_rng(8812311)
    obs_data = []

    xx = np.zeros((n_cells, n_genes), dtype=float)

    for i_cell in range(n_cells):
        this = {
            'cell_label': f'cell_{i_cell}'
        }
        cluster = rng.choice(['cl0', 'cl1', 'cl2', 'cl3'])

        gene_idx = int(cluster[-1])
        vec = 5.0*np.ones(n_genes, dtype=float)
        g0 = gene_idx*10
        vec[g0:g0+5] = rng.random(5)
        vec[g0+5:g0+10] = 7.0+5.0*rng.random(5)
        xx[i_cell, :] = vec

        parentage = taxonomy_tree.parents(
            level='cluster_name_label_alias',
            node=cluster)
        this['cluster_name_label_alias'] = cluster
        for level in parentage:
            this[level] = parentage[level]
        obs_data.append(this)
    obs_df = pd.DataFrame(obs_data).set_index('cell_label')

    var_df = pd.DataFrame(
        [{'gene_id': f'g_{ii}'} for ii in range(n_genes)]
    ).set_index('gene_id')

    a_data = anndata.AnnData(
        X=xx,
        obs=obs_df,
        var=var_df
    )
    a_data.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope='module')
def precomputed_stats_fixture(
        tmp_dir_fixture,
        taxonomy_tree_data_fixture,
        reference_h5ad_fixture):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_stats_',
        suffix='.h5ad'
    )

    precompute_summary_stats_from_h5ad(
        data_path=reference_h5ad_fixture,
        column_hierarchy=['class_name_label_alias',
                          'subclass_name_label_alias',
                          'cluster_name_label_alias'],
        taxonomy_tree=None,
        output_path=h5ad_path,
        tmp_dir=tmp_dir_fixture,
        n_processors=1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tree = TaxonomyTree(data=taxonomy_tree_data_fixture)

    with h5py.File(h5ad_path, 'a') as src:
        del src['taxonomy_tree']
        src.create_dataset(
            'taxonomy_tree',
            data=tree.to_str(drop_cells=True).encode('utf-8')
        )

    return h5ad_path


@pytest.fixture(scope='module')
def marker_genes_fixture(
        tmp_dir_fixture,
        n_genes_fixture,
        taxonomy_tree_data_fixture):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tree = TaxonomyTree(data=taxonomy_tree_data_fixture)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_genes_',
        suffix='.json'
    )

    gene_list = [f'g_{ii}' for ii in range(n_genes_fixture)]

    result = dict()
    rng = np.random.default_rng(838123)
    result['None'] = list(rng.choice(gene_list, 5, replace=False))
    for level in tree.hierarchy:
        for node in tree.nodes_at_level(level):
            k = f'{level}/{node}'
            result[k] = list(rng.choice(gene_list, 10, replace=False))
    with open(output_path, 'w') as dst:
        dst.write(json.dumps(result, indent=2))
    return output_path


@pytest.fixture(scope='module')
def query_h5ad_fixture(
        tmp_dir_fixture,
        n_genes_fixture):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_',
        suffix='.h5ad'
    )

    rng = np.random.default_rng(711221)

    n_cells = 100

    var_df = pd.DataFrame(
        [{'gene_id': f'g_{ii}'} for ii in range(n_genes_fixture)]
    ).set_index('gene_id')
    obs_df = pd.DataFrame(
        [{'cell_id': f'c_{ii}'} for ii in range(n_cells)]
    ).set_index('cell_id')
    a_data = anndata.AnnData(
        X=rng.random((n_cells, n_genes_fixture)),
        obs=obs_df,
        var=var_df
    )
    a_data.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.mark.parametrize(
    'bootstrap_iteration,verbose_csv,mode',
    itertools.product([1, 10], [True, False], ['otf', 'from_spec'])
)
def test_csv_column_names(
        precomputed_stats_fixture,
        marker_genes_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        bootstrap_iteration,
        verbose_csv,
        taxonomy_tree_data_fixture,
        mode):

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csv_output_',
        suffix='.csv'
    )

    json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csv_output_',
        suffix='.json'
    )

    config = {
        'precomputed_stats': {
            'path': precomputed_stats_fixture
        },
        'type_assignment': {
            'bootstrap_iteration': bootstrap_iteration,
            'n_processors': 3,
            'normalization': 'log2CPM',
            'chunk_size': 100
        },
        'csv_result_path': csv_path,
        'extended_result_path': json_path,
        'query_path': query_h5ad_fixture,
        'verbose_csv': verbose_csv
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if mode == 'from_spec':
            config['query_markers'] = {
                'serialized_lookup': marker_genes_fixture
            }
            runner = FromSpecifiedMarkersRunner(
                args=[],
                input_data=config
            )
        elif mode == 'otf':
            config['query_markers'] = {}
            config['reference_markers'] = {}
            config['n_processors'] = (
                config['type_assignment'].pop('n_processors')
            )
            runner = OnTheFlyMapper(args=[], input_data=config)

        runner.run()

    actual_df = pd.read_csv(csv_path, comment='#')

    if verbose_csv:
        metric_suffixes = [
            'bootstrapping_probability',
            'correlation_coefficient',
            'aggregate_probability'
        ]
    else:
        if bootstrap_iteration > 1:
            metric_suffixes = ['bootstrapping_probability']
        else:
            metric_suffixes = ['correlation_coefficient']

    all_suffixes = ['name', 'label'] + metric_suffixes

    expected_columns = set()
    expected_columns.add('cell_id')
    for level in taxonomy_tree_data_fixture['hierarchy']:
        for suffix in all_suffixes:
            expected_columns.add(f'{level}_{suffix}')
    leaf_level = taxonomy_tree_data_fixture['hierarchy'][-1]
    expected_columns.add(f'{leaf_level}_alias')
    actual_columns = set(actual_df.columns)

    missing_columns = []
    for col in expected_columns:
        if col not in actual_columns:
            missing_columns.append(col)
    if len(missing_columns) > 0:
        raise RuntimeError(
            f'columns\n{missing_columns}\nnot in CSV'
        )

    assert expected_columns == actual_columns
