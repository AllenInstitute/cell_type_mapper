"""
Run the full pipeline, testing a case where we know what
clusters cells should be assigned to
"""

import pytest

import anndata
import copy
import h5py
import json
import numpy as np
import os
import pandas as pd
import pathlib
import tempfile

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    use_torch)

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers)

from cell_type_mapper.type_assignment.election import (
    run_type_assignment_on_h5ad_cpu)

from cell_type_mapper.type_assignment.election_runner import (
    run_type_assignment_on_h5ad)

from cell_type_mapper.cli.hierarchical_mapping import (
    run_mapping)

if is_torch_available():
    from cell_type_mapper.gpu_utils.type_assignment.election import (
        run_type_assignment_on_h5ad_gpu)


@pytest.fixture(scope='module')
def precomputed_path_fixture(
        tmp_dir_fixture,
        raw_reference_h5ad_fixture,
        taxonomy_tree_dict):

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)

    precomputed_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    precompute_summary_stats_from_h5ad(
        data_path=raw_reference_h5ad_fixture,
        column_hierarchy=None,
        taxonomy_tree=taxonomy_tree,
        output_path=precomputed_path,
        rows_at_a_time=1000,
        normalization='raw')

    # make sure it is not empty
    with h5py.File(precomputed_path, 'r') as in_file:
        assert len(in_file.keys()) > 0

    return precomputed_path

@pytest.fixture(scope='module')
def ref_marker_path_fixture(
        tmp_dir_fixture,
        precomputed_path_fixture,
        taxonomy_tree_dict):

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)

    ref_marker_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_markers_',
        suffix='.h5')

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precomputed_path_fixture,
        taxonomy_tree=taxonomy_tree,
        output_path=ref_marker_path,
        tmp_dir=tmp_dir_fixture,
        max_bytes=6*1024)

    with h5py.File(ref_marker_path, 'r') as in_file:
        assert len(in_file.keys()) > 0
        assert in_file['up_regulated/data'][()].sum() > 0
        assert in_file['markers/data'][()].sum() > 0

    return ref_marker_path

@pytest.fixture(scope='module')
def marker_cache_path_fixture(
        tmp_dir_fixture,
        taxonomy_tree_dict,
        ref_marker_path_fixture,
        query_gene_names):

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)

    marker_cache_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='ref_and_query_markers_',
        suffix='.h5')

    create_marker_cache_from_reference_markers(
        output_cache_path=marker_cache_path,
        input_cache_path=ref_marker_path_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=1000000)

    with h5py.File(marker_cache_path, 'r') as in_file:
        assert len(in_file['None']['reference'][()]) > 0

    return marker_cache_path


@pytest.mark.parametrize('use_buffer_dir', [True, False])
def test_raw_pipeline(
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        precomputed_path_fixture,
        ref_marker_path_fixture,
        marker_cache_path_fixture,
        use_buffer_dir):

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)

    if use_buffer_dir:
        buffer_dir = tempfile.mkdtemp(
            dir=tmp_dir_fixture,
            prefix='result_buffer_')
    else:
        buffer_dir = None

    result = run_type_assignment_on_h5ad(
        query_h5ad_path=raw_query_h5ad_fixture,
        precomputed_stats_path=precomputed_path_fixture,
        marker_gene_cache_path=marker_cache_path_fixture,
        taxonomy_tree=taxonomy_tree,
        n_processors=3,
        chunk_size=100,
        bootstrap_factor=6.0/7.0,
        bootstrap_iteration=100,
        rng=np.random.default_rng(123545),
        normalization='raw',
        results_output_path=buffer_dir)

    assert len(result) == len(expected_cluster_fixture)
    for cell in result:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]


@pytest.mark.parametrize('use_buffer_dir', [True, False])
def test_raw_pipeline_cpu(
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        precomputed_path_fixture,
        ref_marker_path_fixture,
        marker_cache_path_fixture,
        use_buffer_dir):

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)

    if use_buffer_dir:
        buffer_dir = tempfile.mkdtemp(
            dir=tmp_dir_fixture,
            prefix='result_buffer_')
    else:
        buffer_dir = None

    result = run_type_assignment_on_h5ad_cpu(
        query_h5ad_path=raw_query_h5ad_fixture,
        precomputed_stats_path=precomputed_path_fixture,
        marker_gene_cache_path=marker_cache_path_fixture,
        taxonomy_tree=taxonomy_tree,
        n_processors=3,
        chunk_size=100,
        bootstrap_factor=6.0/7.0,
        bootstrap_iteration=100,
        rng=np.random.default_rng(123545),
        normalization='raw',
        results_output_path=buffer_dir)

    assert len(result) == len(expected_cluster_fixture)
    for cell in result:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]


@pytest.mark.skipif(not is_torch_available(), reason='no torch')
@pytest.mark.parametrize('use_buffer_dir', [True, False])
def test_raw_pipeline_gpu(
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        precomputed_path_fixture,
        ref_marker_path_fixture,
        marker_cache_path_fixture,
        use_buffer_dir):

    if use_buffer_dir:
        buffer_dir = tempfile.mkdtemp(
            dir=tmp_dir_fixture,
            prefix='result_buffer_')
    else:
        buffer_dir = None

    env_var = 'AIBS_BKP_USE_TORCH'
    os.environ[env_var] = 'true'
    assert use_torch()

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)

    result = run_type_assignment_on_h5ad_gpu(
        query_h5ad_path=raw_query_h5ad_fixture,
        precomputed_stats_path=precomputed_path_fixture,
        marker_gene_cache_path=marker_cache_path_fixture,
        taxonomy_tree=taxonomy_tree,
        n_processors=3,
        chunk_size=100,
        bootstrap_factor=6.0/7.0,
        bootstrap_iteration=100,
        rng=np.random.default_rng(123545),
        normalization='raw',
        results_output_path=buffer_dir)

    os.environ[env_var] = ''
    assert not use_torch()

    assert len(result) == len(expected_cluster_fixture)
    for cell in result:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]


@pytest.mark.parametrize(
        'use_tree, check_markers',
        [(True, True),
         (False, False)])
def test_cli_pipeline(
        raw_reference_h5ad_fixture,
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        query_gene_names,
        tmp_dir_fixture,
        use_tree,
        check_markers):

    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture)

    to_store = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture))

    precompute_out = to_store / 'precomputed.h5'
    ref_marker_out = to_store / 'ref_markers.h5'

    config = dict()
    config['tmp_dir'] = tmp_dir
    config['query_path'] = str(
        raw_query_h5ad_fixture.resolve().absolute())

    config['precomputed_stats'] = {
        'reference_path': str(raw_reference_h5ad_fixture.resolve().absolute()),
        'path': str(precompute_out),
        'normalization': 'raw'}

    if use_tree:
        tree_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.json')
        with open(tree_path, 'w') as out_file:
            out_file.write(json.dumps(taxonomy_tree_dict))
        config['precomputed_stats']['taxonomy_tree'] = tree_path
    else:
        config['precomputed_stats']['column_hierarchy'] = taxonomy_tree_dict['hierarchy']

    config['reference_markers'] = {
        'n_processors': 3,
        'max_bytes': 6*1024**2,
        'path': str(ref_marker_out)}

    config["query_markers"] = {
        'n_per_utility': 5,
        'n_processors': 3}

    config["type_assignment"] = {
        'n_processors': 3,
        'bootstrap_factor': 0.9,
        'bootstrap_iteration': 27,
        'rng_seed': 66234,
        'chunk_size': 1000,
        'normalization': 'raw'}

    assert not precompute_out.is_file()
    assert not ref_marker_out.is_file()

    log_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    output_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    run_mapping(
        config,
        output_path=output_path,
        log_path=log_path)

    assert precompute_out.is_file()
    assert ref_marker_out.is_file()

    # check for existence of marker summary
    with h5py.File(ref_marker_out, 'r') as in_file:
        for k in ('sparse_by_pair/up_gene_idx',
                  'sparse_by_pair/up_pair_idx',
                  'sparse_by_pair/down_gene_idx',
                  'sparse_by_pair/down_pair_idx'):
            assert k in in_file
            assert len(in_file[k][()]) > 0

    with open(log_path, 'r') as src:
        log = src.readlines()
    assert len(log) > 0

    results = json.load(open(output_path, 'rb'))
    other_log = results["log"]

    # this is convoluted because the logger as
    # implemented prepends some timing information
    # to the log messages
    for msg in ("creating precomputed stats",
                "creating reference marker file"):
        for this_log in (log, other_log):
            found_it = False
            for line in this_log:
                if msg in line:
                    found_it = True
                    break
            assert found_it

    assert len(results["results"]) == len(expected_cluster_fixture)

    for cell in results["results"]:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]

    # ======== now run it, reusing the precomputed files =========
    config.pop('precomputed_stats')
    config.pop('reference_markers')
    precompute_str = str(precompute_out.resolve().absolute())
    ref_marker_str = str(ref_marker_out.resolve().absolute())

    config['precomputed_stats'] = {'path': precompute_str}
    config['reference_markers'] = {'path': ref_marker_str}

    log_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    output_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    run_mapping(
        config,
        output_path=output_path,
        log_path=log_path)

    with open(log_path, 'r') as src:
        log = src.readlines()
    assert isinstance(log, list)
    assert len(log) > 0

    results = json.load(open(output_path, 'rb'))
    other_log = results["log"]

    # make sure we did not create new stats/marker files
    # when we did not have to
    for msg in ("creating precomputed stats",
                "creating reference marker file"):
        for this_log in (log, other_log):
            found_it = False
            for line in this_log:
                if msg in line:
                    found_it = True
                    break
            assert not found_it

    assert len(results["results"]) == len(expected_cluster_fixture)

    for cell in results["results"]:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]

    # test existence of marker gene lookup
    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)
    assert len(results["marker_genes"]) == len(taxonomy_tree.all_parents)
    for parent in taxonomy_tree.all_parents:
        if parent is None:
            parent_key = 'None'
        else:
            parent_key = f'{parent[0]}/{parent[1]}'
        assert len(results["marker_genes"][parent_key]) > 0

    # check that marker genes were correctly recorded
    # in the JSONized output
    if check_markers:

        query_marker_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5')

        taxonomy_tree=TaxonomyTree(data=taxonomy_tree_dict)

        create_marker_cache_from_reference_markers(
            output_cache_path=query_marker_path,
            input_cache_path=ref_marker_out,
            query_gene_names=query_gene_names,
            taxonomy_tree=taxonomy_tree,
            n_per_utility=config['query_markers']['n_per_utility'],
            n_processors=3,
            behemoth_cutoff=5000000)

        ct = 0
        with h5py.File(query_marker_path, 'r') as baseline:
            full_gene_names = json.loads(
                baseline['reference_gene_names'][()].decode('utf-8'))
            for key in results["marker_genes"]:
                expected = [full_gene_names[ii]
                            for ii in baseline[key]["reference"][()]]
                assert set(expected) == set(results["marker_genes"][key])
                ct += 1
        assert ct == len(taxonomy_tree.all_parents)
        for parent in taxonomy_tree.all_parents:
            if parent is None:
                parent_k = 'None'
            else:
                parent_k = f'{parent[0]}/{parent[1]}'
            assert parent_k in results["marker_genes"]

    # make sure reference file still exists
    assert raw_reference_h5ad_fixture.is_file()


def test_cli_error_log(
        raw_reference_h5ad_fixture,
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        query_gene_names,
        tmp_dir_fixture):
    """
    Same as test_cli_pipeline except configured to fail so that we can
    check the log and make sure the error was captured
    """
    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture)

    to_store = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture))

    ref_marker_out = to_store / 'ref_markers.h5'


    # this will be a bad path
    precompute_out = '/nonexsistent/directory/precomputed.h5'

    config = dict()
    config['tmp_dir'] = tmp_dir
    config['query_path'] = str(
        raw_query_h5ad_fixture.resolve().absolute())

    config['precomputed_stats'] = {
        'reference_path': str(raw_reference_h5ad_fixture.resolve().absolute()),
        'path': precompute_out,
        'normalization': 'raw'}

    tree_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.json')
    with open(tree_path, 'w') as out_file:
            out_file.write(json.dumps(taxonomy_tree_dict))
    config['precomputed_stats']['taxonomy_tree'] = tree_path

    config['reference_markers'] = {
        'n_processors': 3,
        'max_bytes': 6*1024**2,
        'path': str(ref_marker_out)}

    config["query_markers"] = {
        'n_per_utility': 5,
        'n_processors': 3}

    config["type_assignment"] = {
        'n_processors': 3,
        'bootstrap_factor': 0.9,
        'bootstrap_iteration': 27,
        'rng_seed': 66234,
        'chunk_size': 1000,
        'normalization': 'raw'}

    log_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    output_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    with pytest.raises(Exception):
        run_mapping(
            config,
            output_path=output_path,
            log_path=log_path)

    with open(log_path, 'r') as log:
        found_error = False
        found_clean = False
        for line in log:
            if 'an ERROR occurred ====' in line:
                found_error = True
            if 'CLEANING UP' in line:
                found_clean = True
    assert found_error
    assert found_clean
