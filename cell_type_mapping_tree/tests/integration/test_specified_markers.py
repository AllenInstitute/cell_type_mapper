import pytest

import anndata
import copy
import h5py
import json
import numpy as np
import pandas as pd
import pathlib
import tempfile

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers,
    serialize_markers)

from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad)

from hierarchical_mapping.cli.hierarchical_mapping import (
    run_mapping as ab_initio_mapping)

from hierarchical_mapping.cli.from_specified_markers import (
    run_mapping as from_marker_run_mapping)

from hierarchical_mapping.cli.from_specified_markers import (
    FromSpecifiedMarkersRunner)


@pytest.fixture
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


def test_mapping_from_markers(
        ab_initio_assignment_fixture,
        raw_query_cell_x_gene_fixture,
        tmp_dir_fixture):

    this_tmp = tempfile.mkdtemp(dir=tmp_dir_fixture)
    result_path = mkstemp_clean(
        dir=this_tmp,
        suffix='.json')

    baseline_config = ab_initio_assignment_fixture['ab_initio_config']
    config = dict()
    config['tmp_dir'] = this_tmp
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

    config['query_markers'] = {
        'serialized_lookup': ab_initio_assignment_fixture['markers']}

    config['result_path'] = result_path
    config['max_gb'] = 1.0

    runner = FromSpecifiedMarkersRunner(
        args= [],
        input_data=config)

    runner.run()

    actual = json.load(open(result_path, 'rb'))
    expected = json.load(
        open(ab_initio_assignment_fixture['assignment'], 'rb'))

    assert actual['marker_genes'] == expected['marker_genes']
    actual_lookup = {
        cell['cell_id']:cell for cell in actual['results']}
    for cell in expected['results']:
        assert cell == actual_lookup[cell['cell_id']]
    assert len(actual['results']) == raw_query_cell_x_gene_fixture.shape[0]
    assert len(actual['results']) == len(expected['results'])

    # make sure reference file still exists
    assert pathlib.Path(config['precomputed_stats']['reference_path']).is_file()
