"""
Tests to test the workflow as implemented in the online MapMyCells app
"""
import pytest

import anndata
import io
import itertools
import json
import numpy as np
import pandas as pd
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.test_utils.anndata_utils import (
    create_h5ad_without_encoding_type,
    write_anndata_x_to_csv
)

from cell_type_mapper.test_utils.hierarchical_mapping import (
    assert_mappings_equal
)

from cell_type_mapper.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)

from cell_type_mapper.cli.validate_h5ad import (
    ValidateH5adRunner)


def run_pipeline(
        query_path,
        marker_lookup_path,
        precomputed_path,
        tmp_dir,
        gene_mapper_db_path):
    """
    Run the full validation-through-mapping pipeline
    for the online WMB MapMyCells implementation

    Parameters
    ----------
    query_path:
        Path to the input, unmapped file
    marker_lookup_path:
        Path to marker lookup file
    precomputed_path:
        Path to precomputed_stats file
    tmp_dir:
        Path to tmp_dir
    gene_mapper_db_path:
        path to db used by mmc_gene_mapper

    Returns
    --------
    json_path:
        path to JSON output file
    csv_path:
        path to CSV output file
    metadata_path:
        path to summary metadata file
    log_path:
        path to the log file
    """

    log_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='log_',
        suffix='.txt'
    )

    validated_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='validated_',
        suffix='.h5ad')

    output_json_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='output_',
        suffix='.json')

    validation_config = {
        'input_path': query_path,
        'valid_h5ad_path': validated_path,
        'output_json': output_json_path,
        'log_path': log_path,
        'gene_mapping': {
            'db_path': gene_mapper_db_path
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        runner = ValidateH5adRunner(
            args=[],
            input_data=validation_config)
        runner.run()

    output_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='outptut_',
        suffix='.json')

    csv_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='csv_output_',
        suffix='.csv')

    metadata_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='summary_',
        suffix='.json')

    config = {
        'precomputed_stats': {
            'path': str(precomputed_path)
        },
        'query_markers': {
            'serialized_lookup': str(marker_lookup_path)
        },
        'gene_mapping': {
            'db_path': gene_mapper_db_path
        },
        'query_path': validated_path,
        'extended_result_path': str(output_path),
        'csv_result_path': str(csv_path),
        'summary_metadata_path': metadata_path,
        'log_path': log_path,
        'type_assignment': {
            'normalization': 'raw',
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

    return (
        output_path,
        csv_path,
        metadata_path,
        log_path
    )


@pytest.mark.parametrize(
    'with_encoding_type, density_fixture, file_type',
    [(True, 'dense', '.h5ad'),],
    indirect=['density_fixture']
)
def test_online_WMB_log(
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        density_fixture,
        with_encoding_type,
        reference_mapping_fixture,
        file_type,
        legacy_gene_mapper_db_path_fixture
        ):
    """
    Test that .txt log and .json log agree
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

    (json_path,
     _,
     _,
     log_path) = run_pipeline(
         query_path=query_path,
         marker_lookup_path=marker_lookup_fixture,
         precomputed_path=precomputed_stats_fixture,
         tmp_dir=tmp_dir_fixture,
         gene_mapper_db_path=legacy_gene_mapper_db_path_fixture
     )

    with open(log_path, 'r') as src:
        txt_log = src.readlines()

    # scan for 'DONE VALIDATING', indicating that
    # log from validation step was preserved
    found_validation = False
    for line in txt_log:
        if 'DONE VALIDATING' in line:
            found_validation = True
            break
    assert found_validation

    # write JSON log to/from iostream so that any \n
    # in log lines are formatted the same way they are
    # formatted in txt_log
    with open(json_path, 'rb') as src:
        mapping = json.load(src)
    log_stream = io.StringIO()
    for line in mapping['log']:
        log_stream.write(line+'\n')
    log_stream.seek(0)
    json_log = log_stream.readlines()
    assert len(set(json_log)-set(txt_log)) == 0


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
    indirect=['density_fixture']
)
def test_online_workflow_WMB(
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        density_fixture,
        with_encoding_type,
        reference_mapping_fixture,
        file_type,
        legacy_gene_mapper_db_path_fixture
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

    (json_path,
     csv_path,
     _,
     _) = run_pipeline(
         query_path=query_path,
         marker_lookup_path=marker_lookup_fixture,
         precomputed_path=precomputed_stats_fixture,
         tmp_dir=tmp_dir_fixture,
         gene_mapper_db_path=legacy_gene_mapper_db_path_fixture
     )

    test_df = pd.read_csv(
        csv_path,
        comment='#')

    baseline_df = pd.read_csv(
        reference_mapping_fixture['csv_path'],
        comment='#')

    pd.testing.assert_frame_equal(test_df, baseline_df)

    test_dict = json.load(open(json_path, 'rb'))
    baseline_dict = json.load(
        open(reference_mapping_fixture['json_path'], 'rb')
    )
    assert_mappings_equal(
        test_dict['results'],
        baseline_dict['results'],
        compare_cell_id=True)


@pytest.mark.parametrize(
    "cell_label_header,cell_label_type,suffix",
    itertools.product(
        [False, True],
        [None, 'string', 'numerical', 'big_numerical'],
        ['.csv', '.csv.gz']
    )
)
def test_online_workflow_WMB_csv_shape(
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        reference_mapping_fixture,
        cell_label_header,
        cell_label_type,
        suffix,
        legacy_gene_mapper_db_path_fixture
        ):
    """
    Test that mapping is unaffected by the "shape"
    of the CSV file (e.g. whether or not it has a cell_label
    column, etc.)
    """

    if cell_label_header:
        if cell_label_type is None:
            return

    query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_as_csv_',
        suffix=suffix
    )

    write_anndata_x_to_csv(
        anndata_obj=anndata.read_h5ad(query_h5ad_fixture, backed='r'),
        dst_path=query_path,
        cell_label_header=cell_label_header,
        cell_label_type=cell_label_type
    )

    (json_path,
     csv_path,
     _,
     _) = run_pipeline(
         query_path=query_path,
         marker_lookup_path=marker_lookup_fixture,
         precomputed_path=precomputed_stats_fixture,
         tmp_dir=tmp_dir_fixture,
         gene_mapper_db_path=legacy_gene_mapper_db_path_fixture
     )

    test_df = pd.read_csv(
        csv_path,
        comment='#')

    baseline_df = pd.read_csv(
        reference_mapping_fixture['csv_path'],
        comment='#')

    compare_cell_id = False
    if cell_label_type is not None:
        if cell_label_type == 'string':
            compare_cell_id = True

    if not compare_cell_id:
        test_df.drop(['cell_id'], axis='columns', inplace=True)
        baseline_df.drop(['cell_id'], axis='columns', inplace=True)
    pd.testing.assert_frame_equal(test_df, baseline_df)

    test_dict = json.load(open(json_path, 'rb'))
    baseline_dict = json.load(
        open(reference_mapping_fixture['json_path'], 'rb')
    )
    assert_mappings_equal(
        test_dict['results'],
        baseline_dict['results'],
        compare_cell_id=compare_cell_id
    )


def test_online_workflow_WMB_degenerate_cell_labels(
        marker_lookup_fixture,
        precomputed_stats_fixture,
        query_h5ad_fixture,
        tmp_dir_fixture,
        legacy_gene_mapper_db_path_fixture):
    """
    Test that, when cell labels are repeated, the mapping proceeds and
    the order of cells is preserved
    """

    # Create an h5ad file with the same data as
    # query_h5ad_fixture, except that the row pairs
    # specified below in degenerate_pairs have identical
    # cell labels
    degenerate_pairs = [
        (14, 23),
        (7, 111),
        (35, 210)
    ]

    test_h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='with_degenerate_labels_',
        suffix='.h5ad'
    )
    src = anndata.read_h5ad(
        query_h5ad_fixture,
        backed='r')

    src_obs = src.obs
    index_name = src.obs.index.name
    src_obs = src_obs.reset_index().to_dict(orient='records')

    degenerate_idx = set()
    expected_label_lookup = dict()
    for i_pair, pair in enumerate(degenerate_pairs):
        label = f'degeneracy_{i_pair}'
        src_obs[pair[0]][index_name] = label
        src_obs[pair[1]][index_name] = label
        degenerate_idx.add(pair[0])
        degenerate_idx.add(pair[1])
        for idx in pair:
            expected_label_lookup[idx] = (
                '{"cell_id": '
                f'"{label}", "row": {idx}'
                '}'
            )

    new_obs = pd.DataFrame(src_obs).set_index(index_name)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        dst = anndata.AnnData(
            obs=new_obs,
            var=src.var,
            X=src.X
        )
    dst.write_h5ad(test_h5ad_path)
    src.file.close()
    del src

    # do the mapping on the h5ad file with non-degenerate
    # cell labels
    (baseline_json,
     baseline_csv,
     _,
     _) = run_pipeline(
         query_path=query_h5ad_fixture,
         marker_lookup_path=marker_lookup_fixture,
         precomputed_path=precomputed_stats_fixture,
         tmp_dir=tmp_dir_fixture,
         gene_mapper_db_path=legacy_gene_mapper_db_path_fixture
     )

    # do the mapping on the h5ad with degenerate cell labels
    (test_json,
     test_csv,
     _,
     _) = run_pipeline(
         query_path=test_h5ad_path,
         marker_lookup_path=marker_lookup_fixture,
         precomputed_path=precomputed_stats_fixture,
         tmp_dir=tmp_dir_fixture,
         gene_mapper_db_path=legacy_gene_mapper_db_path_fixture
     )

    # compare the contents of the two mappings
    baseline_df = pd.read_csv(
        baseline_csv, comment='#').to_dict(orient='records')

    test_df = pd.read_csv(
        test_csv, comment='#').to_dict(orient='records')

    baseline_mapping = json.load(open(baseline_json, 'rb'))['results']
    test_mapping = json.load(open(test_json, 'rb'))['results']

    assert len(baseline_df) == len(test_df)
    assert len(baseline_mapping) == len(baseline_df)
    assert len(test_mapping) == len(baseline_df)

    for idx in range(len(baseline_df)):
        b_df = baseline_df[idx]
        t_df = test_df[idx]
        b_m = baseline_mapping[idx]
        t_m = test_mapping[idx]
        if idx in degenerate_idx:
            _ = b_df.pop(index_name)
            test_name = t_df.pop(index_name)
            assert test_name == expected_label_lookup[idx]
            test_name = json.loads(test_name)
            assert test_name['row'] == idx
            _ = b_m.pop('cell_id')
            test_name = t_m.pop('cell_id')
            assert test_name == expected_label_lookup[idx]
            test_name = json.loads(test_name)
            assert test_name['row'] == idx
        assert b_df == t_df
        assert b_m == t_m

    # make sure the degenerate cells did not accidentally
    # have identical mappings
    for pair in degenerate_pairs:
        assert index_name not in baseline_df[pair[0]]
        assert index_name not in baseline_df[pair[1]]
        assert baseline_df[pair[0]] != baseline_df[pair[1]]
        assert 'cell_id' not in baseline_mapping[pair[0]]
        assert 'cell_id' not in baseline_mapping[pair[1]]
        assert baseline_mapping[pair[0]] != baseline_mapping[pair[1]]


def test_online_workflow_WMB_perfect_csvs(
        marker_lookup_fixture,
        precomputed_stats_fixture,
        tmp_dir_fixture,
        legacy_gene_mapper_db_path_fixture):
    """
    Test that if CSV has integer counts and ENSEMBL IDs, it can still
    be run (there was a bug in the validation pipeline that caused
    the code not to properly track the h5ad file we were writing out)
    """
    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='perfect_mouse_',
        suffix='.csv'
    )
    marker_set = set()
    with open(marker_lookup_fixture, 'rb') as src:
        markers = json.load(src)
    for key in markers:
        if key in ('log', 'metadata'):
            continue
        marker_set = marker_set.union(set(markers[key]))
    marker_set = sorted(marker_set)
    rng = np.random.default_rng(213131)
    chosen_markers = rng.choice(marker_set, 20, replace=False)
    n_cells = 5
    with open(csv_path, 'w') as dst:
        for gene in chosen_markers:
            dst.write(f',{gene}')
        dst.write('\n')
        for ii in range(n_cells):
            dst.write(f'cell{ii}')
            for jj in range(len(chosen_markers)):
                dst.write(f',{rng.integers(0, 10)}')
            dst.write('\n')
    run_pipeline(
        query_path=csv_path,
        marker_lookup_path=marker_lookup_fixture,
        precomputed_path=precomputed_stats_fixture,
        tmp_dir=tmp_dir_fixture,
        gene_mapper_db_path=legacy_gene_mapper_db_path_fixture
    )
