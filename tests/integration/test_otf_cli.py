"""
tests for the CLI tool that maps to markers which are calculated on
the fly
"""
import pytest

import anndata
import copy
import h5py
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse
import shutil
import tempfile

from cell_type_mapper.data.human_gene_id_lookup import (
    human_gene_id_lookup)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad
)

from cell_type_mapper.test_utils.anndata_utils import (
    create_h5ad_without_encoding_type,
    write_anndata_x_to_csv
)

from cell_type_mapper.utils.output_utils import (
    hdf5_to_blob
)

from cell_type_mapper.diff_exp.precompute_utils import (
    drop_nodes_from_precomputed_stats
)

from cell_type_mapper.test_utils.cloud_safe import (
    check_not_file)

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.cli.reference_markers import (
    ReferenceMarkerRunner)

from cell_type_mapper.cli.query_markers import (
    QueryMarkerRunner)

from cell_type_mapper.cli.map_to_on_the_fly_markers import (
    OnTheFlyMapper)

from cell_type_mapper.cli.validate_h5ad import (
    ValidateH5adRunner)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('otf_test_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def noisy_query_cell_x_gene_fixture(
        cluster_to_signal,
        expected_cluster_fixture,
        query_gene_names):
    """
    Create a noisier cell-by-gene file for querying
    (so we get a mapping with bootstrapping_probability < 1)
    """
    n_cells = len(expected_cluster_fixture)
    n_genes = len(query_gene_names)
    x_data = np.zeros((n_cells, n_genes), dtype=float)
    rng = np.random.default_rng(665533)
    cluster_names = list(set(expected_cluster_fixture))
    cluster_names.sort()
    for i_cell in range(n_cells):
        cl = expected_cluster_fixture[i_cell]
        other_cl = rng.choice(cluster_names)

        signal_lookup = cluster_to_signal[cl]
        other_signal = cluster_to_signal[other_cl]

        signal_amp = (2.0+rng.random())
        alt_amp = (1.0+rng.random())

        data = np.zeros(n_genes, dtype=int)
        noise_idx = rng.choice(np.arange(n_genes), n_genes//3, replace=False)
        data[noise_idx] = rng.integers(1, 50, len(noise_idx))

        for i_gene, g in enumerate(query_gene_names):
            if g in signal_lookup:
                data[i_gene] += signal_amp*signal_lookup[g]
            if g in other_signal:
                data[i_gene] += alt_amp*other_signal[g]

        x_data[i_cell, :] = data

    return np.round(x_data).astype(int)


@pytest.fixture(scope='module')
def noisy_query_h5ad_fixture(
        noisy_query_cell_x_gene_fixture,
        query_gene_names,
        tmp_dir_fixture):
    """
    Create a noisier query dataset
    (so we get a mapping with bootstrapping_probability < 1)
    """
    var_data = [
        {'gene_name': g, 'garbage': ii}
         for ii, g in enumerate(query_gene_names)]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    a_data = anndata.AnnData(
        X=noisy_query_cell_x_gene_fixture,
        var=var,
        uns={'AIBS_CDM_gene_mapping': {'a': 'b', 'c': 'd'}},
        dtype=noisy_query_cell_x_gene_fixture.dtype)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope='module')
def baseline_mapping_fixture(
        precomputed_path_fixture,
        noisy_query_h5ad_fixture,
        tmp_dir_fixture):

    json_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='baseline_',
        suffix='.json'
    )

    csv_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='baseline_',
        suffix='.csv'
    )

    config = {
        'n_processors': 3,
        'tmp_dir': str(tmp_dir_fixture),
        'precomputed_stats': {'path': str(precomputed_path_fixture)},
        'drop_level': None,
        'query_path': str(noisy_query_h5ad_fixture),
        'query_markers': {},
        'reference_markers': {},
        'type_assignment': {
            'normalization': 'raw',
            'rng_seed': 777,
            'bootstrap_factor': 0.5},
        'extended_result_path': json_output,
        'csv_result_path': csv_output,
        'summary_metadata_path': None,
        'cloud_safe': True,
        'nodes_to_drop': None
    }

    runner = OnTheFlyMapper(
        args=[],
        input_data=config
    )
    runner.run()

    return {
        'json': json_output,
        'csv': csv_output
    }


@pytest.fixture(scope='module')
def human_gene_data_fixture(
        precomputed_path_fixture,
        noisy_query_h5ad_fixture,
        tmp_dir_fixture):
    """
    Take the cartoon fixtures created in conftest.py
    and modify them to have valid human gene labels
    so that we can test the full validation-through-mapping
    pipeline.
    """

    rng = np.random.default_rng(6177112)
    genes_to_map = set()
    with h5py.File(precomputed_path_fixture, 'r') as src:
        genes_to_map = genes_to_map.union(
            set(json.loads(src['col_names'][()].decode('utf-8')))
        )
    var = read_df_from_h5ad(
        noisy_query_h5ad_fixture,
        df_name='var')

    genes_to_map = genes_to_map.union(
        set(var.index.values)
    )

    genes_to_map = list(genes_to_map)
    genes_to_map.sort()

    chosen_labels = rng.choice(
        list(human_gene_id_lookup.keys()),
        len(genes_to_map),
        replace=False
    )
    gene_map = {
        g:m for g, m in zip(genes_to_map, chosen_labels)
    }

    new_precompute = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_human_',
        suffix='.h5'
    )

    shutil.copy(src=precomputed_path_fixture, dst=new_precompute)
    with h5py.File(new_precompute, 'a') as dst:
        old_genes = json.loads(dst['col_names'][()].decode('utf-8'))
        del dst['col_names']
        new_genes = [
            human_gene_id_lookup[gene_map[g]] for g in old_genes
        ]
        assert len(new_genes) == len(set(new_genes))
        dst.create_dataset(
            'col_names',
            data=json.dumps(new_genes).encode('utf-8')
        )

    src = anndata.read_h5ad(noisy_query_h5ad_fixture)
    new_var = [
        {'gene_id': gene_map[g]}
        for g in src.var.index.values
    ]
    new_var = pd.DataFrame(new_var).set_index('gene_id')

    result = {'precompute': new_precompute}
    for density in ('csc', 'csr', 'dense'):
        x = src.X[()]
        if density == 'csc':
            x = scipy.sparse.csc_matrix(x)
        elif density == 'csr':
            x = scipy.sparse.csr_matrix(x)
        dst_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix=f'human_query_{density}_',
            suffix='.h5ad'
        )

        dst = anndata.AnnData(
            obs=src.obs,
            var=new_var,
            X=x
        )
        dst.write_h5ad(dst_path)
        result[density] = dst_path

    return result


def test_query_pipeline(
        tmp_dir_fixture,
        precomputed_path_fixture):
    """
    Test that daisy chaining together reference and query marker
    finding produces a result, regardless of accuracy.

    This is to test that the requisite file paths are recorded
    in the metadata of the various intermediate outputs.
    """
    output_dir = pathlib.Path(tempfile.mkdtemp(dir=tmp_dir_fixture))
    assert len([n for n in output_dir.iterdir()]) == 0

    reference_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_markers_',
        suffix='.h5')
    query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_markers_',
        suffix='.json')

    reference_config = {
        'precomputed_path_list': [str(precomputed_path_fixture)],
        'tmp_dir': str(tmp_dir_fixture),
        'n_processors': 3,
        'max_gb': 10,
        'output_dir': str(output_dir)
    }

    ref_runner = ReferenceMarkerRunner(
        args=[],
        input_data=reference_config)
    ref_runner.run()

    assert len([n for n in output_dir.iterdir()]) == 1
    ref_path = [n for n in output_dir.iterdir()][0]

    query_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_markers_',
        suffix='.json')

    query_config = {
        'reference_marker_path_list': [str(ref_path)],
        'output_path': str(query_path),
        'n_processors': 3,
        'tmp_dir': str(tmp_dir_fixture)
    }

    query_runner = QueryMarkerRunner(
        args=[],
        input_data=query_config)
    query_runner.run()

    with open(query_path, 'rb') as src:
        markers = json.load(src)
    assert isinstance(markers, dict)



@pytest.mark.parametrize(
    'write_summary, cloud_safe, keep_encoding',
    itertools.product(
        [True, False],
        [True, False],
        [True, False]))
def test_otf_smoke(
        tmp_dir_fixture,
        precomputed_path_fixture,
        raw_query_h5ad_fixture,
        write_summary,
        cloud_safe,
        keep_encoding):

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='mapping_',
        suffix='.json')

    if write_summary:
        metadata_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='summary_metadata_',
            suffix='.json')
    else:
        metadata_path = None

    if keep_encoding:
        src_path = str(raw_query_h5ad_fixture)
    else:
        src_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='no_encoding_',
            suffix='.h5ad'
        )
        create_h5ad_without_encoding_type(
            src_path=raw_query_h5ad_fixture,
            dst_path=src_path
        )

    config = {
        'n_processors': 3,
        'tmp_dir': tmp_dir,
        'precomputed_stats': {'path': str(precomputed_path_fixture)},
        'drop_level': None,
        'query_path': src_path,
        'query_markers': {},
        'reference_markers': {},
        'type_assignment': {'normalization': 'raw'},
        'extended_result_path': output_path,
        'summary_metadata_path': metadata_path,
        'cloud_safe': cloud_safe,
        'nodes_to_drop': None
    }

    runner = OnTheFlyMapper(args=[], input_data=config)
    runner.run()

    result = json.load(open(output_path, 'rb'))
    assert 'RAN SUCCESSFULLY' in result['log'][-2]
    assert 'marker_genes' in result
    assert len(result['marker_genes']) > 3

    raw_data = anndata.read_h5ad(raw_query_h5ad_fixture, backed='r')
    n_cells = len(raw_data.obs)
    assert len(result['results']) == n_cells

    if write_summary:
        metadata = json.load(open(metadata_path, 'rb'))
        assert metadata['n_mapped_cells'] == n_cells
        n_genes = len(read_df_from_h5ad(raw_query_h5ad_fixture, df_name='var'))
        assert metadata['n_mapped_genes'] == n_genes

    if cloud_safe:
        with open(output_path, 'rb') as src:
            data = json.load(src)
        check_not_file(data['config'])
        check_not_file(data['log'])


def test_otf_no_markers(
        tmp_dir_fixture,
        precomputed_path_fixture):
    """
    Check that the correct error is raised when reference marker finding
    fails.
    """

    query_path = mkstemp_clean(
       dir=tmp_dir_fixture,
       suffix='.h5ad')

    n_genes = 10
    n_cells = 15
    var = pd.DataFrame(
        [{'gene_id': f'garbage_{ii}'}
         for ii in range(n_genes)]).set_index('gene_id')
    obs = pd.DataFrame(
        [{'cell_id': f'c_{ii}'}
         for ii in range(n_cells)]).set_index('cell_id')
    rng = np.random.default_rng(5513)
    x = rng.integers(0, 255, (n_cells, n_genes))
    src = anndata.AnnData(X=x, obs=obs, var=var)
    src.write_h5ad(query_path)

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='mapping_',
        suffix='.json')

    metadata_path = None

    config = {
        'n_processors': 3,
        'tmp_dir': tmp_dir,
        'precomputed_stats': {'path': str(precomputed_path_fixture)},
        'drop_level': None,
        'query_path': query_path,
        'query_markers': {},
        'reference_markers': {},
        'type_assignment': {'normalization': 'raw'},
        'extended_result_path': output_path,
        'summary_metadata_path': metadata_path
    }

    runner = OnTheFlyMapper(args=[], input_data=config)
    msg = (
        "Genes in query data file do not overlap genes in "
        "reference data file."
    )
    with pytest.raises(RuntimeError, match=msg):
        runner.run()



@pytest.mark.parametrize(
    "nodes_to_drop",
    [
     [('class', 'a'), ('subclass', 'subclass_5')],
     [('class', 'a'), ('class', 'b')]
    ]
)
def test_otf_drop_nodes(
        tmp_dir_fixture,
        precomputed_path_fixture,
        raw_query_h5ad_fixture,
        nodes_to_drop):
    """
    Run on-the-fly mapping, once on the full precomputed_stats
    file, calling nodes_to_drop. Once on a precomputed_stats file
    that was pre-munged. Compare that results are identical.
    """

    # record hash of precomputed stats file to make sure it
    # is not changed when nodes are dropped
    hasher = hashlib.md5()
    with open(precomputed_path_fixture, 'rb') as src:
        hasher.update(src.read())
    precompute_hash = hasher.hexdigest()

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    baseline_output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='baseline_mapping_',
        suffix='.json')

    pre_munged_output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='pre_munged_mapping_',
        suffix='.json')

    drop_nodes_output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='drop_nodes_mapping_',
        suffix='.json')

    munged_stats_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='pre_munged_',
        suffix='.h5'
    )

    drop_nodes_from_precomputed_stats(
        src_path=precomputed_path_fixture,
        dst_path=munged_stats_path,
        node_list=nodes_to_drop,
        clobber=True
    )

    for output_path, precompute_path, drop_nodes in [
            (baseline_output_path, precomputed_path_fixture, None),
            (pre_munged_output_path, munged_stats_path, None),
            (drop_nodes_output_path, precomputed_path_fixture, nodes_to_drop)]:

        config = {
            'n_processors': 3,
            'tmp_dir': tmp_dir,
            'precomputed_stats': {'path': str(precompute_path)},
            'drop_level': None,
            'query_path': str(raw_query_h5ad_fixture),
            'query_markers': {},
            'reference_markers': {},
            'type_assignment': {'normalization': 'raw'},
            'extended_result_path': output_path,
            'summary_metadata_path': None,
            'cloud_safe': False,
            'nodes_to_drop': drop_nodes
        }

        runner = OnTheFlyMapper(args=[], input_data=config)
        runner.run()

    hasher = hashlib.md5()
    with open(precomputed_path_fixture, 'rb') as src:
        hasher.update(src.read())
    final_hash = hasher.hexdigest()
    assert final_hash == precompute_hash

    baseline = json.load(open(baseline_output_path, 'rb'))
    munged = json.load(open(pre_munged_output_path, 'rb'))
    dropped = json.load(open(drop_nodes_output_path, 'rb'))

    assert munged['results'] != baseline['results']
    assert munged['results'] == dropped['results']

    assert munged['marker_genes'] != baseline['marker_genes']
    assert munged['marker_genes'] == dropped['marker_genes']

    assert munged['taxonomy_tree'] != baseline['taxonomy_tree']
    assert munged['taxonomy_tree'] == dropped['taxonomy_tree']




@pytest.mark.parametrize(
    "nodes_to_drop",
    [None,
     [('class', 'a'), ('subclass', 'subclass_5')]]
)
def test_otf_config_consistency(
        tmp_dir_fixture,
        noisier_precomputed_path_fixture,
        raw_query_h5ad_fixture,
        nodes_to_drop):
    """
    Test that you can just pass the config file from the mapping
    result JSON back into the module, and get the same result.
    """

    hdf5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5'
    )
    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    base_config = {
        'n_processors': 3,
        'tmp_dir': this_tmp,
        'precomputed_stats': {'path': str(noisier_precomputed_path_fixture)},
        'drop_level': None,
        'query_path': str(raw_query_h5ad_fixture),
        'query_markers': {
            'n_per_utility': 30
        },
        'reference_markers': {
            'log2_fold_min_th': 0.8,
            'q1_th': 0.5,
            'q1_min_th': 0.1,
            'qdiff_min_th': 0.1
        },
        'type_assignment': {
            'normalization': 'raw',
            'rng_seed': 11235,
            'bootstrap_factor': 0.4},
        'summary_metadata_path': None,
        'cloud_safe': False,
        'nodes_to_drop': nodes_to_drop,
        'hdf5_result_path': hdf5_path
    }

    baseline_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a_',
        suffix='.json'
    )

    base_config['extended_result_path'] = baseline_output

    runner = OnTheFlyMapper(args=[], input_data=base_config)
    runner.run()
    baseline_mapping = json.load(open(baseline_output, 'rb'))

    blob = hdf5_to_blob(hdf5_path)
    assert blob['config'] == baseline_mapping['config']
    assert blob['metadata'] == baseline_mapping['metadata']

    test_config = copy.deepcopy(baseline_mapping['config'])
    test_output = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a2_',
        suffix='.json'
    )
    test_config['extended_result_path'] = test_output
    runner = OnTheFlyMapper(args=[], input_data=test_config)
    runner.run()
    test_mapping = json.load(open(test_output, 'rb'))
    for k in ('results', 'marker_genes', 'taxonomy_tree'):
        assert test_mapping[k] == baseline_mapping[k]

    update_config_list = [
        {'n_processors': 2},
        {'type_assignment': {'rng_seed': 566122}},
        {'reference_markers': {
                'log2_fold_min_th': 0.9,
                'q1_th': 0.9,
                'q1_min_th': 0.8,
                'qdiff_min_th': 0.5
            }
        },
        {'query_markers': {'n_per_utility': 5}}

    ]
    for update_config in update_config_list:
        test_config = copy.deepcopy(base_config)
        test_output = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.json'
        )
        test_config['extended_result_path'] = test_output

        # change test_config parametrs
        for k in update_config:
            if not isinstance(update_config[k], dict):
                assert k in test_config
                test_config[k] = update_config[k]
            else:
                for k2 in update_config[k]:
                    assert k2 in test_config[k]
                    test_config[k][k2] = update_config[k][k2]

        runner = OnTheFlyMapper(args=[], input_data=test_config)
        runner.run()
        test_mapping = json.load(open(test_output, 'rb'))
        blob = hdf5_to_blob(test_config['hdf5_result_path'])
        assert blob['config'] == test_mapping['config']
        assert blob['metadata'] == test_mapping['metadata']

        # make sure result changed where expected
        assert test_mapping['results'] != baseline_mapping['results']
        assert test_mapping['taxonomy_tree'] == baseline_mapping['taxonomy_tree']
        if 'reference_markers' in update_config or 'query_markers' in update_config:
            assert test_mapping['marker_genes'] != baseline_mapping['marker_genes']
        else:
            assert test_mapping['marker_genes'] == baseline_mapping['marker_genes']

        # Make sure test_mapping recorded a config that allows you to
        # reproduce its results
        second_test_config = copy.deepcopy(test_mapping['config'])
        second_test_output = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.json'
        )
        second_test_config['extended_result_path'] = second_test_output
        runner = OnTheFlyMapper(args=[], input_data=second_test_config)
        runner.run()
        second_test_mapping = json.load(open(second_test_output, 'rb'))
        for k in ('results', 'taxonomy_tree', 'marker_genes'):
            assert second_test_mapping[k] == test_mapping[k]


@pytest.mark.parametrize(
    'keep_encoding,density,file_type',
    [(True, 'dense', '.h5ad'),
     (False, 'dense', '.h5ad'),
     (True, 'csr', '.h5ad'),
     (False, 'csr', '.h5ad'),
     (True, 'csc', '.h5ad'),
     (False, 'csc', '.h5ad'),
     (True, 'dense', '.csv.gz'),
     (True, 'dense', '.csv')])
def test_online_workflow_OTF(
        tmp_dir_fixture,
        keep_encoding,
        human_gene_data_fixture,
        density,
        baseline_mapping_fixture,
        file_type):
    """
    Test the validation-through-mapping flow of simulated human
    data passing through the on-the-fly mapper
    """

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    precompute_path = human_gene_data_fixture['precompute']
    query_path = human_gene_data_fixture[density]

    if file_type == '.h5ad':
        if not keep_encoding:
            new_path = mkstemp_clean(
                dir=tmp_dir,
                prefix='no_encoding_',
                suffix='.h5ad'
            )
            create_h5ad_without_encoding_type(
                src_path=query_path,
                dst_path=new_path
            )
            query_path = new_path
    else:
        new_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='query_as_csv_',
            suffix=file_type
        )
        write_anndata_x_to_csv(
            anndata_obj=anndata.read_h5ad(query_path, backed='r'),
            dst_path=new_path
        )
        query_path = new_path

    validated_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='validated_',
        suffix='.h5ad'
    )

    output_json = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='validation_output_',
        suffix='.json'
    )

    validation_config = {
        'h5ad_path': query_path,
        'valid_h5ad_path': validated_path,
        'output_json': output_json
    }

    runner = ValidateH5adRunner(
        args=[],
        input_data=validation_config
    )
    runner.run()

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='mapping_',
        suffix='.json')

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='csv_mapping_',
        suffix='.csv'
    )

    metadata_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='summary_metadata_',
            suffix='.json')

    config = {
        'n_processors': 3,
        'tmp_dir': tmp_dir,
        'precomputed_stats': {'path': str(precompute_path)},
        'drop_level': None,
        'query_path': validated_path,
        'query_markers': {},
        'reference_markers': {},
        'type_assignment': {
            'normalization': 'raw',
            'rng_seed': 777,
            'bootstrap_factor': 0.5},
        'extended_result_path': output_path,
        'csv_result_path': csv_path,
        'summary_metadata_path': metadata_path,
        'cloud_safe': True,
        'nodes_to_drop': None
    }

    runner = OnTheFlyMapper(args=[], input_data=config)
    runner.run()

    baseline_df = pd.read_csv(
        baseline_mapping_fixture['csv'],
        comment='#'
    )

    test_df = pd.read_csv(
        csv_path,
        comment='#'
    )

    pd.testing.assert_frame_equal(test_df, baseline_df)

    baseline = json.load(open(baseline_mapping_fixture['json'], 'rb'))
    test = json.load(open(output_path, 'rb'))
    assert baseline['results'] == test['results']
