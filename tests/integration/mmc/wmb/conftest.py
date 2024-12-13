import pytest

import anndata
import copy
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse
import shutil

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.test_utils.anndata_utils import (
    create_h5ad_without_encoding_type,
    write_anndata_x_to_csv
)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.diff_exp.truncate_precompute import (
    truncate_precomputed_stats_file
)

from cell_type_mapper.data.mouse_gene_id_lookup import (
    mouse_gene_id_lookup)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)

from cell_type_mapper.cli.validate_h5ad import (
    ValidateH5adRunner)


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
def n_extra_genes_fixture():
    """
    Number of unmappable genes to include in the data
    """
    return 19

@pytest.fixture()
def density_fixture(request):
    if not hasattr(request, 'param'):
        return 'dense'
    else:
        return request.param



@pytest.fixture()
def query_h5ad_fixture(
        density_fixture,
        gene_name_fixture,
        tmp_dir_fixture,
        n_extra_genes_fixture):

    return create_query_h5ad(
        density_specification=density_fixture,
        gene_name_list=gene_name_fixture,
        tmp_dir_path=tmp_dir_fixture,
        n_extra_genes=n_extra_genes_fixture
    )


def create_query_h5ad(
        density_specification,
        gene_name_list,
        tmp_dir_path,
        n_extra_genes):

    if not hasattr(create_query_h5ad, '_cache'):
        create_query_h5ad._cache = dict()

    if density_specification not in create_query_h5ad._cache:
        h5ad_path = mkstemp_clean(
            dir=tmp_dir_path,
            prefix='query_data_',
            suffix='.h5ad')

        n_cells = 321
        n_genes = len(gene_name_list) + n_extra_genes

        rng = np.random.default_rng(77123)

        if density_specification == 'dense':
            X = rng.random((n_cells, n_genes), dtype=np.float32)
        else:
            n_tot = n_cells*n_genes
            data = np.zeros(n_tot, dtype=int)
            chosen_idx = rng.choice(n_tot, n_tot//3, replace=False)
            data[chosen_idx] = rng.integers(1, 255, len(chosen_idx))
            data = data.reshape((n_cells, n_genes))
            if density_specification == 'csc':
                X = scipy.sparse.csc_matrix(data)
            elif density_specification == 'csr':
                X = scipy.sparse.csr_matrix(data)
            else:
                raise RuntimeError(
                    f'cannot parse density {density_specification}'
                )

        obs = pd.DataFrame(
            [{'cell_id': f'cell_{ii}'}
             for ii in range(n_cells)]).set_index('cell_id')

        these_gene_names = copy.deepcopy(gene_name_list)
        for ii in range(n_extra_genes):
            these_gene_names.append(f'extra_gene_{ii}')
        rng.shuffle(these_gene_names)

        var = pd.DataFrame(
            [{'gene_name': g} for g in these_gene_names]).set_index('gene_name')

        a_data = anndata.AnnData(
            X=X,
            obs=obs,
            var=var)

        a_data.write_h5ad(h5ad_path)
        create_query_h5ad._cache[density_specification] = h5ad_path
    return create_query_h5ad._cache[density_specification]


@pytest.fixture()
def reference_mapping_fixture(
        query_h5ad_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        density_fixture,
        tmp_dir_fixture):
    """
    A fixture returning what the mapping should be
    (for comparing results when we are mapping data
    that lacked the encoding-type metadata field)
    """
    return do_reference_mapping(
        query_h5ad_path=query_h5ad_fixture,
        marker_lookup_path=marker_lookup_fixture,
        precomputed_stats_path=precomputed_stats_fixture,
        density_specification=density_fixture,
        tmp_dir_path=tmp_dir_fixture
    )


def do_reference_mapping(
        query_h5ad_path,
        marker_lookup_path,
        precomputed_stats_path,
        density_specification,
        tmp_dir_path):

    if not hasattr(do_reference_mapping, '_cache'):
        do_reference_mapping._cache = dict()

    if density_specification not in do_reference_mapping._cache:
        validated_path = mkstemp_clean(
            dir=tmp_dir_path,
            prefix='validated_',
            suffix='.h5ad')

        output_json_path = mkstemp_clean(
            dir=tmp_dir_path,
            prefix='output_',
            suffix='.json')

        validation_config = {
            'h5ad_path': str(query_h5ad_path),
            'valid_h5ad_path': validated_path,
            'output_json': output_json_path}

        runner = ValidateH5adRunner(
            args=[],
            input_data=validation_config)
        runner.run()

        output_path = mkstemp_clean(
            dir=tmp_dir_path,
            prefix='outptut_',
            suffix='.json')

        csv_path = mkstemp_clean(
            dir=tmp_dir_path,
            prefix='csv_output_',
            suffix='.csv')

        metadata_path = mkstemp_clean(
            dir=tmp_dir_path,
            prefix='summary_',
            suffix='.json')

        config = {
            'precomputed_stats': {
                'path': str(precomputed_stats_path)
            },
            'query_markers': {
                'serialized_lookup': str(marker_lookup_path)
            },
            'query_path': validated_path,
            'extended_result_path': str(output_path),
            'csv_result_path': str(csv_path),
            'summary_metadata_path': metadata_path,
            'map_to_ensembl': False,
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

        runner.run()

        this = {
            'csv_path': csv_path,
            'json_path': output_path
        }
        do_reference_mapping._cache[density_specification] = this

    return do_reference_mapping._cache[density_specification]
