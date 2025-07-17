"""
Implement some smoketest-level tests of the from_specified_markers
CLI tool.
"""
import pytest

import anndata
import itertools
import json
import numpy as np
import pandas as pd
import shutil
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.diff_exp.truncate_precompute import (
    truncate_precomputed_stats_file
)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)

from cell_type_mapper.cli.validate_h5ad import (
    ValidateH5adRunner)


@pytest.mark.parametrize(
    'map_to_ensembl,write_summary',
    itertools.product([True, False], [True, False])
)
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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
                assert metadata['n_mapped_genes'] == (
                    len(_var) - n_extra_genes_fixture
                )
        else:
            msg = (
                "After comparing query data to reference data, "
                "no valid marker genes could be found"
            )
            with pytest.raises(RuntimeError, match=msg):
                runner.run()


def test_summary_from_validated_file(
        taxonomy_tree_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        n_extra_genes_fixture):
    """
    This test makes sure that the summary metadata is correctly recorded
    when ensembl mapping is handled by the validation CLI.

    Additionally test that cells in the output CSV file are in the same
    order as in the input h5ad file.
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
        'input_path': str(query_h5ad_fixture),
        'valid_h5ad_path': validated_path,
        'output_json': output_json_path}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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
                                          - n_extra_genes_fixture)

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        runner = FromSpecifiedMarkersRunner(
            args=[],
            input_data=config)

        runner.run()
    actual = json.load(open(output_path, 'rb'))
    assert 'RAN SUCCESSFULLY' in actual['log'][-2]
