"""
Unit tests for actually performing gene mapping
across species
"""

import pytest

import anndata
import copy
import itertools
import json
import numpy as np
import pandas as pd

import mmc_gene_mapper.utils.str_utils as str_utils

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.cli.precompute_stats_scrattch import (
    PrecomputationScrattchRunner
)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner
)

from cell_type_mapper.cli.map_to_on_the_fly_markers import (
    OnTheFlyMapper
)

from cell_type_mapper.test_utils.hierarchical_mapping import (
    assert_mappings_equal
)


@pytest.fixture(scope='session')
def precomputed_mouse_fixture(
        tmp_dir_fixture):
    """
    Create an precomputed_stats file defining a dummy taxonomy in
    mouse. Return a path to that file

    classes: CLAS0, CLAS1
    subclasses: SUBC{class#*10}+[0, 1, 2]
    clusters: CLUS{SUBC#*10}+[0, 1]
    """
    n_cells = 960
    rng = np.random.default_rng(213211)
    gene_list = [
        {'gene_id': f'NCBIGene:{ii}'}
        for ii in range(0, 100)
    ]
    var = pd.DataFrame(gene_list).set_index('gene_id')
    n_genes = len(gene_list)
    x_arr = np.zeros((n_cells, n_genes), dtype=int)
    obs = []
    reference_profile = dict()
    for i_cell in range(n_cells):
        cl_idx = rng.integers(0, 2)
        subcl_idx = 10*cl_idx + rng.integers(0, 3)
        clus_idx = 10*subcl_idx + rng.integers(0, 2)
        class_label = f'CLAS{cl_idx}'
        subclass_label = f'SUBC{subcl_idx}'
        cluster_label = f'CLUS{clus_idx}'
        obs.append(
            {'class': class_label,
             'subclass': subclass_label,
             'cluster': cluster_label,
             'cell_id': f'c_{i_cell}'}
        )
        if cluster_label not in reference_profile:
            chosen_genes = rng.choice(np.arange(n_genes), 25, replace=False)
            vals = rng.integers(5, 25, len(chosen_genes))
            profile = np.zeros(n_genes, dtype=int)
            profile[chosen_genes] = vals
            reference_profile[cluster_label] = profile
        chosen_idx = rng.choice(np.arange(n_genes), 50, replace=False)
        cell_profile = np.zeros(n_genes, dtype=int)
        cell_profile[chosen_idx] = rng.integers(1, 4, len(chosen_idx))
        cell_profile += rng.integers(1, 3)*reference_profile[cluster_label]
        x_arr[i_cell, :] = cell_profile

    obs = pd.DataFrame(obs).set_index('cell_id')

    a_data = anndata.AnnData(
        obs=obs,
        var=var,
        X=x_arr
    )

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='ait_mouse_',
        suffix='.h5ad'
    )

    a_data.write_h5ad(h5ad_path)

    precompute_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='mouse_taxonomy_for_orthologs_',
        suffix='.h5'
    )

    config = {
        'output_path': str(precompute_path),
        'h5ad_path': str(h5ad_path),
        'hierarchy': ['class', 'subclass', 'cluster'],
        'normalization': 'raw',
        'clobber': True,
        'tmp_dir': str(tmp_dir_fixture)
    }
    runner = PrecomputationScrattchRunner(
        args=[],
        input_data=config
    )
    runner.run()
    return str(precompute_path)


@pytest.fixture(scope='module')
def query_markers_fixture(tmp_dir_fixture):
    """
    Return path to JSON file listing query markers
    """
    rng = np.random.default_rng(22313131)
    json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json'
    )
    markers = dict()
    all_int = np.arange(100, dtype=int)
    markers['None'] = [
        f'NCBIGene:{ii}'
        for ii in rng.choice(all_int, 30, replace=False)
    ]
    for ii in range(2):
        key = f'class/CLAS{ii}'
        markers[key] = [
            f'NCBIGene:{nn}'
            for nn in rng.choice(all_int, 25, replace=False)
        ]
        for jj in range(3):
            key = f'subclass/SUBC{ii*10+jj}'
            markers[key] = [
                f'NCBIGene:{nn}'
                for nn in rng.choice(all_int, 23, replace=False)
            ]
            for kk in range(2):
                key = f'cluster/CLUS{ii*100+jj*19+kk}'
                markers[key] = [
                    f'NCBIGene:{nn}'
                    for nn in rng.choice(all_int, 41, replace=False)
                ]
    with open(json_path, 'w') as dst:
        dst.write(json.dumps(markers, indent=2))
    return str(json_path)


@pytest.mark.parametrize(
     'authority, mapper_mode',
     itertools.product(
         ['NCBI', 'ENSEMBL'],
         ['specified', 'otf']
     )
 )
def test_mapping_with_orthologs(
        tmp_dir_fixture,
        precomputed_mouse_fixture,
        query_markers_fixture,
        mapper_db_path_fixture,
        authority,
        mapper_mode):
    """
    Generate a query dataset whose genes are identified
    by mouse NCBI genes (just like in reference set).

    Generate a query dataset whose genes are identified
    by human genes identified according to the specified
    authority.

    Map them both.

    Make sure mappings are equivalent.

    Make sure human mapping output carries correct gene_identifier_mapping
    metadata.
    """

    rng = np.random.default_rng(771231)

    query_gene_list = []
    baseline_gene_list = []
    for ii in range(0, 100, 2):
        mouse_gene = f'NCBIGene:{ii}'
        if authority == 'NCBI':
            gene = f'NCBIGene:{100+ii}'
        else:
            gene = f'ENSG{100+ii}'
        baseline_gene_list.append({'gene': mouse_gene})
        query_gene_list.append({'gene': gene})

    # add genes that cannot be mapped
    for ii in range(9):
        gene = f'garbage:{1000+ii}'
        query_gene_list.append({'gene': gene})
        baseline_gene_list.append({'gene': gene})

    baseline_var = pd.DataFrame(baseline_gene_list).set_index('gene')
    query_var = pd.DataFrame(query_gene_list).set_index('gene')

    obs = pd.DataFrame(
        [{'cell_id': f'query_cell_{ii}'} for ii in range(200)]
    ).set_index('cell_id')

    x_arr = rng.integers(0, 55, (len(obs), len(baseline_var)))

    baseline_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='baseline_',
        suffix='.h5ad'
    )

    query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='test_',
        suffix='.h5ad'
    )

    baseline_a_data = anndata.AnnData(
        var=baseline_var,
        obs=obs,
        X=x_arr)
    baseline_a_data.write_h5ad(baseline_path)

    query_a_data = anndata.AnnData(
        var=query_var,
        obs=obs,
        X=x_arr)
    query_a_data.write_h5ad(query_path)

    rng_seed = 3141592654

    baseline_output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json'
    )

    if mapper_mode == 'specified':
        baseline_config = {
            'query_path': baseline_path,
            'extended_result_path': baseline_output_path,
            'gene_mapping': None,
            'cloud_safe': False,
            'precomputed_stats': {'path': precomputed_mouse_fixture},
            'query_markers': {'serialized_lookup': query_markers_fixture},
            'type_assignment': {
                'n_processors': 2,
                'rng_seed': rng_seed,
                'bootstrap_factor': 0.5,
                'bootstrap_iteration': 100,
                'chunk_size': 50,
                'normalization': 'raw'
            }
        }

        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=baseline_config
        )
        runner.run()
    else:
        baseline_config = {
            'query_path': baseline_path,
            'extended_result_path': baseline_output_path,
            'gene_mapping': None,
            'cloud_safe': False,
            'precomputed_stats': {'path': precomputed_mouse_fixture},
            'n_processors': 2,
            'type_assignment': {
                'rng_seed': rng_seed,
                'bootstrap_factor': 0.5,
                'bootstrap_iteration': 100,
                'chunk_size': 50,
                'normalization': 'raw'
            },
            'tmp_dir': str(tmp_dir_fixture),
            'reference_markers': {
                'n_valid': 10
            },
            'query_markers': {
                'n_per_utility': 5
            }
        }

        runner = OnTheFlyMapper(
            args=[],
            input_data=baseline_config
        )
        runner.run()

    with open(baseline_output_path, 'rb') as src:
        baseline_output = json.load(src)

    query_config = copy.deepcopy(baseline_config)
    query_config['query_path'] = str(query_path)
    query_output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.json'
    )
    query_config['extended_result_path'] = str(query_output_path)
    query_config['gene_mapping'] = {'db_path': mapper_db_path_fixture}

    if mapper_mode == 'specified':
        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=query_config
        )
        runner.run()
    else:
        runner = OnTheFlyMapper(
            args=[],
            input_data=query_config
        )
        runner.run()

    with open(query_output_path, 'rb') as src:
        query_output = json.load(src)

    assert_mappings_equal(
        mapping0=baseline_output['results'],
        mapping1=query_output['results']
    )

    gene_mapping = query_output['gene_identifier_mapping']['mapping']
    for src_gene in gene_mapping:
        src_idx = str_utils.int_from_identifier(src_gene)
        if src_idx < 201:
            assert gene_mapping[src_gene] == f'NCBIGene:{src_idx-100}'
        else:
            assert 'UNMAPPABLE' in gene_mapping[src_gene]
