"""
tests for the CLI tool that maps to markers which are calculated on
the fly
"""
import pytest

import anndata
import copy
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import tempfile
from unittest.mock import patch

from cell_type_mapper.diff_exp.precompute_utils import (
    drop_nodes_from_precomputed_stats
)

from cell_type_mapper.test_utils.cloud_safe import (
    check_not_file)

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.cli.reference_markers import (
    ReferenceMarkerRunner)

from cell_type_mapper.cli.query_markers import (
    QueryMarkerRunner)

from cell_type_mapper.cli.map_to_on_the_fly_markers import (
    OnTheFlyMapper)


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
    'write_summary, cloud_safe',
    itertools.product(
        [True, False],
        [True, False]))
def test_otf_smoke(
        tmp_dir_fixture,
        precomputed_path_fixture,
        raw_query_h5ad_fixture,
        write_summary,
        cloud_safe):

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

    config = {
        'n_processors': 3,
        'tmp_dir': tmp_dir,
        'precomputed_stats': {'path': str(precomputed_path_fixture)},
        'drop_level': None,
        'query_path': str(raw_query_h5ad_fixture),
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
            'rng_seed': 11235},
        'summary_metadata_path': None,
        'cloud_safe': False,
        'nodes_to_drop': nodes_to_drop
    }

    output_a = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a_',
        suffix='.json'
    )
    config_a = copy.deepcopy(base_config)
    config_a['extended_result_path'] = output_a
    config_a['type_assignment']['bootstrap_factor'] = 0.4

    runner = OnTheFlyMapper(args=[], input_data=config_a)
    runner.run()
    mapping_a = json.load(open(output_a, 'rb'))

    output_a2 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a2_',
        suffix='.json'
    )
    config_a2 = copy.deepcopy(mapping_a['config'])
    config_a2['extended_result_path'] = output_a2
    runner = OnTheFlyMapper(args=[], input_data=config_a2)
    runner.run()
    mapping_a2 = json.load(open(output_a2, 'rb'))

    assert mapping_a['results'] == mapping_a2['results']
    assert mapping_a['marker_genes'] == mapping_a2['marker_genes']
    assert mapping_a['taxonomy_tree'] == mapping_a2['taxonomy_tree']

    # change n_processors; make sure mapping changes
    config_a3 = copy.deepcopy(mapping_a['config'])
    output_a3 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a3_',
        suffix='.json'
    )
    config_a3['n_processors'] = 2
    config_a3['extended_result_path'] = output_a3
    runner = OnTheFlyMapper(args=[], input_data=config_a3)
    runner.run()
    mapping_a3 = json.load(open(output_a3, 'rb'))
    assert mapping_a['results'] != mapping_a3['results']
    assert mapping_a['marker_genes'] == mapping_a3['marker_genes']
    assert mapping_a['taxonomy_tree'] == mapping_a3['taxonomy_tree']

    # change rng_seed; make sure mapping changes
    config_a4 = copy.deepcopy(mapping_a['config'])
    output_a4 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a4_',
        suffix='.json'
    )
    config_a4['type_assignment']['rng_seed'] = 55667
    config_a4['extended_result_path'] = output_a4
    runner = OnTheFlyMapper(args=[], input_data=config_a4)
    runner.run()
    mapping_a4 = json.load(open(output_a4, 'rb'))
    assert mapping_a['results'] != mapping_a4['results']
    assert mapping_a['marker_genes'] == mapping_a4['marker_genes']
    assert mapping_a['taxonomy_tree'] == mapping_a4['taxonomy_tree']

    # change reference marker configs
    config_a5 = copy.deepcopy(mapping_a['config'])
    output_a5 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a5_',
        suffix='.json'
    )

    config_a5['reference_markers']['log2_fold_min_th'] = 0.9
    config_a5['reference_markers']['q1_th'] = 0.9
    config_a5['reference_markers']['q1_min_th'] = 0.8
    config_a5['reference_markers']['qdiff_min_th'] = 0.5

    config_a5['extended_result_path'] = output_a5
    runner = OnTheFlyMapper(args=[], input_data=config_a5)
    runner.run()
    mapping_a5 = json.load(open(output_a5, 'rb'))
    assert mapping_a['results'] != mapping_a5['results']
    assert mapping_a['marker_genes'] != mapping_a5['marker_genes']
    assert mapping_a['taxonomy_tree'] == mapping_a5['taxonomy_tree']

    config_a6 = copy.deepcopy(mapping_a5['config'])
    output_a6 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a6_',
        suffix='.json'
    )
    config_a6['extended_result_path'] = output_a6
    runner = OnTheFlyMapper(args=[], input_data=config_a6)
    runner.run()
    mapping_a6 = json.load(open(output_a6, 'rb'))
    assert mapping_a6['results'] == mapping_a5['results']
    assert mapping_a6['marker_genes'] == mapping_a5['marker_genes']
    assert mapping_a6['taxonomy_tree'] == mapping_a5['taxonomy_tree']

    # change query marker configs
    config_a7 = copy.deepcopy(mapping_a['config'])
    output_a7 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a7_',
        suffix='.json'
    )

    config_a7['query_markers']['n_per_utility'] = 5

    config_a7['extended_result_path'] = output_a7
    runner = OnTheFlyMapper(args=[], input_data=config_a7)
    runner.run()
    mapping_a7 = json.load(open(output_a5, 'rb'))
    assert mapping_a['results'] != mapping_a7['results']
    assert mapping_a['marker_genes'] != mapping_a7['marker_genes']
    assert mapping_a['taxonomy_tree'] == mapping_a7['taxonomy_tree']

    config_a8 = copy.deepcopy(mapping_a7['config'])
    output_a8 = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='output_a8_',
        suffix='.json'
    )
    config_a8['extended_result_path'] = output_a8
    runner = OnTheFlyMapper(args=[], input_data=config_a8)
    runner.run()
    mapping_a8 = json.load(open(output_a8, 'rb'))
    assert mapping_a8['results'] == mapping_a7['results']
    assert mapping_a8['marker_genes'] == mapping_a7['marker_genes']
    assert mapping_a8['taxonomy_tree'] == mapping_a7['taxonomy_tree']
