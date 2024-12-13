"""
Tests to test the workflow as implemented in the online MapMyCells app
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


@pytest.mark.parametrize(
        'with_encoding_type, density_fixture, file_type',
        [(True, 'dense', '.h5ad'),
         (False, 'dense', '.h5ad'),
         (True, 'csc', '.h5ad'),
         (False, 'csc', '.h5ad'),
         (True, 'csr', '.h5ad'),
         (False, 'csr', '.h5ad'),
         (True, 'dense', '.csv'),
         (True, 'dense', '.csv.gz')
        ],
        indirect=['density_fixture'])
def test_online_workflow_WMB(
        taxonomy_tree_fixture,
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        n_extra_genes_fixture,
        density_fixture,
        with_encoding_type,
        reference_mapping_fixture,
        file_type
        ):
    """
    Test the validation through mapping workflow as it will be run
    on Whole Mouse Brain data.

    Creating this test especially so that we can verify the functionality
    to patch query data that is missing proper encoding-type metadata
    """
    if file_type == '.h5ad':
        if with_encoding_type:
            query_path = str(query_h5ad_fixture)
        else:
            query_path = mkstemp_clean(
                dir=tmp_dir_fixture,
                prefix='query_without_encoding_type_',
                suffix='.h5ad'
            )
            create_h5ad_without_encoding_type(
                src_path=query_h5ad_fixture,
                dst_path=query_path
            )
    else:
        query_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='query_as_csv_',
            suffix=file_type
        )

        write_anndata_x_to_csv(
            anndata_obj=anndata.read_h5ad(query_h5ad_fixture, backed='r'),
            dst_path=query_path
        )

    validated_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='validated_',
        suffix='.h5ad')

    output_json_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_',
        suffix='.json')

    validation_config = {
        'h5ad_path': query_path,
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

    if not with_encoding_type:
        test_df = pd.read_csv(
            csv_path,
            comment='#')

        baseline_df = pd.read_csv(
            reference_mapping_fixture['csv_path'],
            comment='#')

        pd.testing.assert_frame_equal(test_df, baseline_df)

        test_dict = json.load(open(output_path, 'rb'))
        baseline_dict = json.load(open(reference_mapping_fixture['json_path'], 'rb'))
        assert test_dict['results'] == baseline_dict['results']
