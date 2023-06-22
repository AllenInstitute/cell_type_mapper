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

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.utils.torch_utils import (
    is_torch_available)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers,
    serialize_markers)

from hierarchical_mapping.type_assignment.election_runner import (
    run_type_assignment_on_h5ad)

from hierarchical_mapping.cli.hierarchical_mapping import (
    run_mapping as ab_initio_mapping)

from hierarchical_mapping.cli.from_specified_markers import (
    run_mapping as from_marker_run_mapping)

from hierarchical_mapping.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)


@pytest.fixture(scope='module')
def ab_initio_assignment_fixture(
        raw_reference_h5ad_fixture,
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        query_gene_names,
        tmp_dir_fixture):
    """
    Run the (tested) assignment pipeline.
    Store its results.
    Serialize its markers to a JSON file.
    Will later test that we get back the same results
    using the specified markers.
    """

    this_tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture))

    precompute_out = this_tmp_dir / 'precomputed.h5'
    ref_marker_out = this_tmp_dir / 'ref.h5'

    config = dict()
    config['tmp_dir'] = str(this_tmp_dir.resolve().absolute())
    config['query_path'] = str(
        raw_query_h5ad_fixture.resolve().absolute())

    config['precomputed_stats'] = {
        'reference_path': str(raw_reference_h5ad_fixture.resolve().absolute()),
        'path': str(precompute_out),
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

    assignment_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    ab_initio_mapping(
        config,
        output_path=str(assignment_path),
        log_path=None)

    # create query marker cache to serialize

    query_marker_path = mkstemp_clean(
        dir=this_tmp_dir,
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

    marker_lookup = serialize_markers(
        marker_cache_path=query_marker_path,
        taxonomy_tree=taxonomy_tree)

    marker_lookup_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.json'))

    with open(marker_lookup_path, 'w') as out_file:
        out_file.write(json.dumps(marker_lookup))

    _clean_up(this_tmp_dir)

    yield {'assignment': assignment_path,
           'markers': str(marker_lookup_path.resolve().absolute()),
           'ab_initio_config': config}

    marker_lookup_path.unlink()
    assignment_path.unlink()


@pytest.mark.parametrize(
        'flatten,use_csv,use_tmp_dir,use_gpu',
        [(True, True, True, False),
         (True, False, True, False),
         (False, True, True, False),
         (False, False, True, False),
         (False, True, True, False),
         (False, True, True, True),
         (True, True, True, True)])
def test_mapping_from_markers(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        taxonomy_tree_dict,
        tmp_dir_fixture,
        flatten,
        use_csv,
        use_tmp_dir,
        use_gpu):

    if use_gpu and not is_torch_available():
        return

    env_var = 'AIBS_BKP_USE_TORCH'
    if use_gpu:
        os.environ[env_var] = 'true'
    else:
        os.environ[env_var] = 'false'

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)

    if use_csv:
        csv_path = mkstemp_clean(
            dir=this_tmp,
            suffix='.csv')
    else:
        csv_path = None

    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    baseline_config = ab_initio_assignment_fixture['ab_initio_config']
    config = dict()
    if use_tmp_dir:
        config['tmp_dir'] = this_tmp
    else:
        config['tmp_dir'] = None
    config['query_path'] = baseline_config['query_path']
    config['precomputed_stats'] = copy.deepcopy(baseline_config['precomputed_stats'])
    config['precomputed_stats'].pop('path')

    new_stats_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5',
            prefix='precomputed_stats_',
            delete=True)

    config['precomputed_stats']['path'] = new_stats_path
    config['type_assignment'] = copy.deepcopy(baseline_config['type_assignment'])
    config['flatten'] = flatten

    config['query_markers'] = {
        'serialized_lookup': ab_initio_assignment_fixture['markers']}

    config['extended_result_path'] = result_path
    config['csv_result_path'] = csv_path
    config['max_gb'] = 1.0

    runner = FromSpecifiedMarkersRunner(
        args= [],
        input_data=config)

    runner.run()

    actual = json.load(open(result_path, 'rb'))

    gpu_msg = 'Running GPU implementation of type assignment.'
    cpu_msg = 'Running CPU implementation of type assignment.'
    found_gpu = False
    found_cpu = False
    for line in actual['log']:
        if gpu_msg in line:
            found_gpu = True
        if cpu_msg in line:
            found_cpu = True

    if found_cpu:
        assert not found_gpu
    if found_gpu:
        assert not found_cpu

    if use_gpu:
        assert found_gpu
    else:
        assert found_cpu

    # this is only expectd if flatten == False
    expected = json.load(
        open(ab_initio_assignment_fixture['assignment'], 'rb'))


    if not flatten:
        assert actual['marker_genes'] == expected['marker_genes']
        actual_lookup = {
            cell['cell_id']:cell for cell in actual['results']}
        for cell in expected['results']:
            actual_cell = actual_lookup[cell['cell_id']]
            assert set(cell.keys()) == set(actual_cell.keys())
            for k in cell.keys():
                if k == 'cell_id':
                    continue
                assert set(cell[k].keys()) == set(actual_cell[k].keys())
                assert cell[k]['assignment'] == actual_cell[k]['assignment']
                for sub_k in ('confidence', 'avg_correlation'):
                    np.testing.assert_allclose(
                        [cell[k][sub_k]],
                        [actual_cell[k][sub_k]],
                        atol=1.0e-4,
                        rtol=1.0e-4)
    else:
        all_markers = set()
        for k in expected['marker_genes']:
            all_markers = all_markers.union(set(expected['marker_genes'][k]))

        assert set(actual['marker_genes']['None']) == all_markers
        assert len(actual['marker_genes']['None']) == len(all_markers)
        assert len(actual['marker_genes']) == 1
        valid_clusters = set(taxonomy_tree_dict['cluster'].keys())
        for cell in actual['results']:
            assert 'cluster' in cell
            assert 'assignment' in cell['cluster']
            assert 'confidence' in cell['cluster']
            assert 'avg_correlation' in cell['cluster']
            assert cell['cluster']['assignment'] in valid_clusters

    assert len(actual['results']) == raw_query_cell_x_gene_fixture.shape[0]
    assert len(actual['results']) == len(expected['results'])

    # make sure reference file still exists
    assert pathlib.Path(config['precomputed_stats']['reference_path']).is_file()

    # check consistency between extended and csv results
    if use_csv:
        result_lookup = {
            cell['cell_id']: cell for cell in actual['results']}
        with open(csv_path, 'r') as in_file:
            assert in_file.readline() == f"# metadata = {pathlib.Path(result_path).name}\n"
            if flatten:
                hierarchy = ['cluster']
            else:
                hierarchy = ['class', 'subclass', 'cluster']
            assert in_file.readline() == f"# taxonomy hierarchy = {json.dumps(hierarchy)}\n"

            header_line = 'cell_id'
            for level in hierarchy:
                header_line += f',{level},{level}_confidence'
            header_line += '\n'
            assert in_file.readline() == header_line
            found_cells = []
            for line in in_file:
                params = line.strip().split(',')
                assert len(params) == 2*len(hierarchy)+1
                this_cell = result_lookup[params[0]]
                found_cells.append(params[0])
                for i_level, level in enumerate(hierarchy):
                    assert params[1+2*i_level] == this_cell[level]['assignment']
                    delta = np.abs(this_cell[level]['confidence']-float(params[2+2*i_level]))
                    assert delta < 0.0001

            assert len(found_cells) == len(result_lookup)
            assert set(found_cells) == set(result_lookup.keys())

    os.environ[env_var] = ''
