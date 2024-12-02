"""
Implement some smoketest-level tests of the from_specified_markers
CLI tool.
"""
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

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_data_',
        suffix='.h5ad')

    n_cells = 321
    n_genes = len(gene_name_fixture) + n_extra_genes_fixture

    rng = np.random.default_rng(77123)

    if density_fixture == 'dense':
        X = rng.random((n_cells, n_genes), dtype=np.float32)
    else:
        n_tot = n_cells*n_genes
        data = np.zeros(n_tot, dtype=int)
        chosen_idx = rng.choice(n_tot, n_tot//3, replace=False)
        data[chosen_idx] = rng.integers(1, 255, len(chosen_idx))
        data = data.reshape((n_cells, n_genes))
        if density_fixture == 'csc':
            X = scipy.sparse.csc_matrix(data)
        elif density_fixture == 'csr':
            X = scipy.sparse.csr_matrix(data)
        else:
            raise RuntimeError(
                f'cannot parse density {density_fixture}'
            )

    obs = pd.DataFrame(
        [{'cell_id': f'cell_{ii}'}
         for ii in range(n_cells)]).set_index('cell_id')

    these_gene_names = copy.deepcopy(gene_name_fixture)
    for ii in range(n_extra_genes_fixture):
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


def test_online_workflow_WMB(
        taxonomy_tree_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        n_extra_genes_fixture):
    """
    Test the validation through mapping workflow as it will be run
    on Whole Mouse Brain data.

    Creating this test especially so that we can verify the functionality
    to patch query data that is missing proper encoding-type metadata
    """

    validated_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='validated_',
        suffix='.h5ad')

    output_json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_',
        suffix='.json')

    validation_config = {
        'h5ad_path': str(query_h5ad_fixture),
        'valid_h5ad_path': validated_path,
        'output_json': output_json_path}

    runner = ValidateH5adRunner(
        args=[],
        input_data=validation_config)
    runner.run()

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='outptut_',
        suffix='.json')

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csv_output_',
        suffix='.csv')

    metadata_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='summary_',
        suffix='.json')

    config = {
        'precomputed_stats': {
            'path': str(precomputed_stats_fixture)
        },
        'query_markers': {
            'serialized_lookup': str(marker_lookup_fixture)
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


@pytest.mark.parametrize('map_to_ensembl,write_summary',
    itertools.product([True, False], [True, False]))
def test_ensembl_mapping_in_cli(
        taxonomy_tree_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        n_extra_genes_fixture,
        map_to_ensembl,
        write_summary):
    """
    Test for expected behavior (error/no error) when we just
    ask the from_specified_markers CLI to map gene names to
    ENSEMBLID
    """
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='outptut_',
        suffix='.json')

    if write_summary:
        metadata_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='summary_',
            suffix='.json')
    else:
        metadata_path = None

    config = {
        'precomputed_stats': {
            'path': str(precomputed_stats_fixture)
        },
        'query_markers': {
            'serialized_lookup': str(marker_lookup_fixture)
        },
        'query_path': str(query_h5ad_fixture),
        'extended_result_path': str(output_path),
        'summary_metadata_path': metadata_path,
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
        if write_summary:
            metadata = json.load(open(metadata_path, 'rb'))
            assert 'n_mapped_cells' in metadata
            assert 'n_mapped_genes' in metadata
            _obs = read_df_from_h5ad(query_h5ad_fixture, df_name='obs')
            _var = read_df_from_h5ad(query_h5ad_fixture, df_name='var')
            assert metadata['n_mapped_cells'] == len(_obs)
            assert metadata['n_mapped_genes'] == (len(_var)
                                                  -n_extra_genes_fixture)
    else:
        msg = (
            "After comparing query data to reference data, "
            "no valid marker genes could be found"
        )
        with pytest.raises(RuntimeError, match=msg):
            runner.run()



@pytest.mark.parametrize('map_to_ensembl',
    [True, False])
def test_summary_from_validated_file(
        taxonomy_tree_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        n_extra_genes_fixture,
        map_to_ensembl):
    """
    This test makes sure that the summary metadata is correctly recorded
    when ensembl mapping is handled by the validation CLI.

    Additionally test that cells in the output CSV file are in the same
    order as in the input h5ad file.

    Toggling map_to_ensemble makes sure that the summary metadata
    is correctly recorded, even when the ensembl mapping was handled
    by the validation layer
    """

    validated_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='validated_',
        suffix='.h5ad')

    output_json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_',
        suffix='.json')

    validation_config = {
        'h5ad_path': str(query_h5ad_fixture),
        'valid_h5ad_path': validated_path,
        'output_json': output_json_path}

    runner = ValidateH5adRunner(
        args=[],
        input_data=validation_config)
    runner.run()

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='outptut_',
        suffix='.json')

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csv_output_',
        suffix='.csv')

    metadata_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='summary_',
        suffix='.json')

    config = {
        'precomputed_stats': {
            'path': str(precomputed_stats_fixture)
        },
        'query_markers': {
            'serialized_lookup': str(marker_lookup_fixture)
        },
        'query_path': validated_path,
        'extended_result_path': str(output_path),
        'csv_result_path': str(csv_path),
        'summary_metadata_path': metadata_path,
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

    runner.run()

    metadata = json.load(open(metadata_path, 'rb'))
    assert 'n_mapped_cells' in metadata
    assert 'n_mapped_genes' in metadata

    # need to copy query file into another path
    # otherwise there is a swmr conflict with
    # tests run in parallel
    query_h5ad_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.h5ad')
    shutil.copy(src=query_h5ad_fixture, dst=query_h5ad_path)

    query_data = anndata.read_h5ad(query_h5ad_path, backed='r')
    assert metadata['n_mapped_cells'] == len(query_data.obs)
    assert metadata['n_mapped_genes'] == (len(query_data.var)
                                          -n_extra_genes_fixture)

    src_obs = read_df_from_h5ad(query_h5ad_fixture, df_name='obs')
    mapping_df = pd.read_csv(csv_path, comment='#')
    assert len(mapping_df) == len(src_obs)
    np.testing.assert_array_equal(
        mapping_df.cell_id.values, src_obs.index.values)



@pytest.mark.parametrize(
    'hierarchy',
    [
        ('class',),
        ('class', 'subclass'),
        ('subclass',),
        ('class', 'cluster'),
        ('subclass', 'cluster'),
        ('cluster',)
    ])
def test_cli_on_truncated_precompute(
        taxonomy_tree_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        n_extra_genes_fixture,
        hierarchy):
    """
    Run a smoke test on FromSpecifiedMarkersRunner using a
    precomputed stats file that has been truncated
    """
    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='outptut_',
        suffix='.json')

    metadata_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='summary_',
        suffix='.json')

    new_precompute_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    truncate_precomputed_stats_file(
        input_path=precomputed_stats_fixture,
        output_path=new_precompute_path,
        new_hierarchy=hierarchy)

    config = {
        'precomputed_stats': {
            'path': str(new_precompute_path)
        },
        'query_markers': {
            'serialized_lookup': str(marker_lookup_fixture)
        },
        'query_path': str(query_h5ad_fixture),
        'extended_result_path': str(output_path),
        'summary_metadata_path': metadata_path,
        'map_to_ensembl': True,
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
    actual = json.load(open(output_path, 'rb'))
    assert 'RAN SUCCESSFULLY' in actual['log'][-2]
