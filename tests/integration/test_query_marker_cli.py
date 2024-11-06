"""
Test the CLI tool for finding query markers
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
import tempfile


from cell_type_mapper.test_utils.reference_markers import (
    move_precomputed_stats_from_reference_markers,
    move_precomputed_stats_from_mask_file
)

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.csc_to_csr_parallel import (
    transpose_sparse_matrix_on_disk_v2)

from cell_type_mapper.cli.query_markers import (
    QueryMarkerRunner)

from cell_type_mapper.cli.compute_p_value_mask import (
    PValueRunner)

from cell_type_mapper.cli.query_markers_from_p_value_mask import(
    QueryMarkersFromPValueMaskRunner)


@pytest.mark.parametrize(
    "n_per_utility,drop_level,downsample_genes",
    itertools.product(
        (5, 3, 7, 11),
        (None, 'subclass'),
        (True, False)))
def test_query_marker_cli_tool(
        query_gene_names,
        ref_marker_path_fixture,
        precomputed_path_fixture,
        full_marker_name_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        n_per_utility,
        drop_level,
        downsample_genes):

    if downsample_genes:
        rng = np.random.default_rng(76123)
        valid_gene_names = rng.choice(
            query_gene_names,
            len(query_gene_names)*3//4,
            replace=False)

        query_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='h5ad_for_finding_query_markers_',
            suffix='.h5ad')

        var = pd.DataFrame(
            [{'gene_name': g}
             for g in valid_gene_names]).set_index('gene_name')
        adata = anndata.AnnData(var=var)
        adata.write_h5ad(query_path)
    else:
        valid_gene_names = query_gene_names
        query_path = None

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_markers_',
        suffix='.json')

    config = {
        'query_path': query_path,
        'reference_marker_path_list': [ref_marker_path_fixture],
        'n_processors': 3,
        'n_per_utility': n_per_utility,
        'drop_level': drop_level,
        'output_path': output_path,
        'tmp_dir': str(tmp_dir_fixture.resolve().absolute())}

    runner = QueryMarkerRunner(
        args=[],
        input_data=config)
    runner.run()

    with open(output_path, 'rb') as src:
        actual = json.load(src)

    # test roundtrip of config
    alt_output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='alt_query_markers_',
        suffix='.json'
    )

    new_config = copy.deepcopy(actual['metadata']['config'])
    new_config.pop('output_path')
    new_config['output_path'] = alt_output_path
    new_runner = QueryMarkerRunner(
        args=[],
        input_data=new_config)
    new_runner.run()
    with open(alt_output_path, 'rb') as src:
        roundtrip = json.load(src)
    assert set(roundtrip.keys()) == set(actual.keys())
    for k in actual:
        if k in ('log', 'metadata'):
            continue
        assert roundtrip[k] == actual[k]

    # verify value of contents in actual
    assert 'log' in actual
    n_skipped = 0
    n_dur = 0
    log = actual['log']
    for level in taxonomy_tree_dict['hierarchy'][:-1]:
        for node in taxonomy_tree_dict[level]:
            log_key = f'{level}/{node}'
            if level == drop_level:
                assert log_key not in log
            else:
                assert log_key in log
                is_skipped = False
                if 'msg' in log[log_key]:
                    if 'Skipping; no leaf' in log[log_key]['msg']:
                        is_skipped = True
                if is_skipped:
                    n_skip += 1
                else:
                    assert 'duration' in log[log_key]
                    n_dur += 1

    assert n_dur > 0

    gene_ct = 0
    levels_found = set()
    actual_genes = set()
    for k in actual:
        if k == 'metadata':
            continue
        if k == 'log':
            continue
        if drop_level is not None:
            assert drop_level not in k
        levels_found.add(k.split('/')[0])
        for g in actual[k]:
            actual_genes.add(g)
            assert g in valid_gene_names
            gene_ct += 1
    assert gene_ct > 0

    expected_levels = set(['None'])
    for level in taxonomy_tree_dict['hierarchy'][:-1]:
        if level != drop_level:
            expected_levels.add(level)
    assert expected_levels == levels_found

    if not downsample_genes and n_per_utility == 7 and drop_level is None:
        assert actual_genes == set(full_marker_name_fixture)
    elif downsample_genes:
        assert actual_genes != set(full_marker_name_fixture)

    assert 'metadata' in actual
    assert 'timestamp' in actual['metadata']
    assert 'config' in actual['metadata']
    for k in config:
        assert k in actual['metadata']['config']
        assert actual['metadata']['config'][k] == config[k]


@pytest.mark.parametrize(
    "n_per_utility,drop_level,downsample_genes,search_for_stats_file",
    itertools.product(
        (3, 7),
        (None, 'subclass'),
        (True, False),
        (True, False)))
def test_missing_precompute_query_marker(
        query_gene_names,
        ref_marker_path_fixture,
        precomputed_path_fixture,
        full_marker_name_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        n_per_utility,
        drop_level,
        downsample_genes,
        search_for_stats_file):
    """
    Test that query marker CLI properly handles precomputed_stats files
    whose absolute paths have changed.
    """
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture)
    )

    new_ref_list = move_precomputed_stats_from_reference_markers(
        reference_marker_path_list=[ref_marker_path_fixture],
        tmp_dir=tmp_dir)

    if downsample_genes:
        rng = np.random.default_rng(76123)
        valid_gene_names = rng.choice(
            query_gene_names,
            len(query_gene_names)*3//4,
            replace=False)

        query_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='h5ad_for_finding_query_markers_',
            suffix='.h5ad')

        var = pd.DataFrame(
            [{'gene_name': g}
             for g in valid_gene_names]).set_index('gene_name')
        adata = anndata.AnnData(var=var)
        adata.write_h5ad(query_path)
    else:
        valid_gene_names = query_gene_names
        query_path = None

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_markers_',
        suffix='.json')

    config = {
        'query_path': query_path,
        'reference_marker_path_list': new_ref_list,
        'n_processors': 3,
        'n_per_utility': n_per_utility,
        'drop_level': drop_level,
        'output_path': output_path,
        'tmp_dir': str(tmp_dir_fixture.resolve().absolute()),
        'search_for_stats_file': search_for_stats_file}

    runner = QueryMarkerRunner(
        args=[],
        input_data=config)


    if search_for_stats_file:
        runner.run()
    else:
        match = "Could not find the following precomputed_stats"
        with pytest.raises(FileNotFoundError, match=match):
            runner.run()

    _clean_up(tmp_dir)


def test_transposing_markers(
        ref_marker_path_fixture,
        tmp_dir_fixture):
    """
    Test transposition of sparse array using 'realistic'
    reference marker data.
    """

    src_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    with h5py.File(ref_marker_path_fixture, 'r') as src:
        n_rows = src['n_pairs'][()]
        n_cols = len(json.loads(src['gene_names'][()].decode('utf-8')))
        indices = src['sparse_by_pair/up_gene_idx'][()]
        indptr = src['sparse_by_pair/up_pair_idx'][()]

    data = (indices+1)**2
    csr = scipy.sparse.csr_array(
        (data, indices, indptr),
        shape=(n_rows, n_cols))

    expected_csc = scipy.sparse.csc_array(
        csr.toarray())

    with h5py.File(src_path, 'w') as dst:
        dst.create_dataset('data', data=data, chunks=(1000,))
        dst.create_dataset('indices', data=indices, chunks=(1000,))
        dst.create_dataset('indptr', data=indptr)

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.h5')

    transpose_sparse_matrix_on_disk_v2(
        h5_path=src_path,
        indices_tag='indices',
        indptr_tag='indptr',
        data_tag='data',
        indices_max=n_cols,
        max_gb=1,
        n_processors=3,
        output_path=dst_path,
        verbose=False,
        tmp_dir=tmp_dir_fixture)

    with h5py.File(dst_path, 'r') as src:
        actual_data = src['data'][()]
        actual_indices = src['indices'][()]
        actual_indptr = src['indptr'][()]

    assert actual_indices.shape == expected_csc.indices.shape

    np.testing.assert_array_equal(
        actual_indptr,
        expected_csc.indptr)

    np.testing.assert_array_equal(
        actual_indices[-10:],
        expected_csc.indices[-10:])

    actual_csc = scipy.sparse.csc_matrix(
        (actual_data, actual_indices, actual_indptr),
        shape=(n_rows,n_cols))

    np.testing.assert_array_equal(
        csr.toarray(), actual_csc.toarray())


def test_genes_at_a_time(
        query_gene_names,
        ref_marker_path_fixture,
        precomputed_path_fixture,
        full_marker_name_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture):
    """
    Really just a smoke test to make sure that
    genes_at_a_time changes the result
    """

    valid_gene_names = query_gene_names
    query_path = None

    baseline_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='baseline_query_markers_',
        suffix='.json')

    config = {
        'query_path': query_path,
        'reference_marker_path_list': [ref_marker_path_fixture],
        'n_processors': 3,
        'n_per_utility': 5,
        'drop_level': None,
        'output_path': baseline_path,
        'tmp_dir': str(tmp_dir_fixture.resolve().absolute())}

    runner = QueryMarkerRunner(
        args=[],
        input_data=config)
    runner.run()

    for genes_at_a_time in (1, 10):
        test_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='test_query_markers_',
            suffix='.json')

        config['output_path'] = test_path
        config['genes_at_a_time'] = genes_at_a_time

        runner = QueryMarkerRunner(
            args=[],
            input_data=config)
        runner.run()

        baseline = json.load(open(baseline_path, 'rb'))
        test = json.load(open(test_path, 'rb'))
        for k in ('log', 'metadata'):
            baseline.pop(k)
            test.pop(k)

        if genes_at_a_time == 1:
            assert test == baseline
        else:
            assert test != baseline


@pytest.fixture(scope='module')
def p_value_path_fixture(
        precomputed_path_fixture,
        tmp_dir_fixture):

    p_value_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='p_value_mask_',
        suffix='.h5')

    config = {
        'tmp_dir': str(tmp_dir_fixture),
        'output_path': p_value_path,
        'precomputed_stats_path': precomputed_path_fixture,
        'clobber': True,
        'n_processors': 3
    }

    runner = PValueRunner(
        args=[],
        input_data=config)
    runner.run()
    return p_value_path


@pytest.mark.parametrize(
        "n_per_utility,genes_at_a_time,n_processors,n_valid",
        itertools.product(
            (10,),
            (5,),
            (3,),
            (None,)
        ))
def test_query_markers_from_p_values(
        tmp_dir_fixture,
        p_value_path_fixture,
        genes_at_a_time,
        n_per_utility,
        n_processors,
        n_valid):
    """
    Just a smoke test for the CLI tool that goes straight from
    p-value mask to query markers.
    """

    drop_level = None
    n_per_utility_override = None

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_from_p_values_',
        suffix='.json')

    config = {
        'output_path': output_path,
        'p_value_mask_path': p_value_path_fixture,
        'max_gb': 5,
        'tmp_dir': str(tmp_dir_fixture),
        'n_processors': n_processors,
        'drop_level': drop_level,
        'clobber': True,
        'reference_markers': {
            'n_valid': n_valid
        },
        'query_markers':  {
            'n_per_utility': n_per_utility,
            'n_per_utility_override': n_per_utility_override,
            'genes_at_a_time': genes_at_a_time
        }
    }

    runner = QueryMarkersFromPValueMaskRunner(
        args=[],
        input_data=config)

    runner.run()

    with open(output_path, 'rb') as src:
        result = json.load(src)
    assert len(result) > 2
    n_markers = 0
    for k in result:
        if k in ('log', 'metadata'):
            continue
        n_markers += len(result[k])
    assert n_markers > 0

    # test roundtrip of config
    new_config = copy.deepcopy(result['metadata']['config'])
    new_config.pop('output_path')
    test_output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='roundtrip_query_markers_from_mask_',
        suffix='.json'
    )
    new_config['output_path'] = test_output_path
    new_runner = QueryMarkersFromPValueMaskRunner(
        args=[],
        input_data=new_config)

    new_runner.run()
    with open(test_output_path, 'rb') as src:
        roundtrip = json.load(src)
    assert set(roundtrip.keys()) == set(result.keys())
    for k in roundtrip.keys():
        if k in ('log', 'metadata'):
            continue
        assert roundtrip[k] == result[k]


@pytest.mark.parametrize("search_for_stats", [True, False])
def test_query_markers_from_p_values_when_precompute_moved(
        tmp_dir_fixture,
        p_value_path_fixture,
        search_for_stats):
    """
    Just a smoke test for the CLI tool that goes straight from
    p-value mask to query markers in the case where the precomputed_stats
    file has been moved
    """
    n_per_utility = 10
    genes_at_a_time = 3
    n_valid = None
    n_processors = 3


    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_fixture)

    new_mask_path = move_precomputed_stats_from_mask_file(
        mask_file_path=p_value_path_fixture,
        tmp_dir=tmp_dir)

    drop_level = None
    n_per_utility_override = None

    output_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='query_from_p_values_',
        suffix='.json')

    config = {
        'output_path': output_path,
        'p_value_mask_path': str(new_mask_path.resolve().absolute()),
        'max_gb': 5,
        'tmp_dir': str(tmp_dir_fixture),
        'n_processors': n_processors,
        'drop_level': drop_level,
        'clobber': True,
        'reference_markers': {
            'n_valid': n_valid
        },
        'query_markers':  {
            'n_per_utility': n_per_utility,
            'n_per_utility_override': n_per_utility_override,
            'genes_at_a_time': genes_at_a_time
        },
        'search_for_stats_file': search_for_stats
    }

    runner = QueryMarkersFromPValueMaskRunner(
        args=[],
        input_data=config)

    if not search_for_stats:
        match = "and saving the missing file"
        with pytest.raises(FileNotFoundError, match=match):
            runner.run()
    else:

        runner.run()

        with open(output_path, 'rb') as src:
            result = json.load(src)
        assert len(result) > 2
        n_markers = 0
        for k in result:
            if k in ('log', 'metadata'):
                continue
            n_markers += len(result[k])
        assert n_markers > 0

    _clean_up(tmp_dir)
