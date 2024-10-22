"""
tests for the CLI tool that maps to markers which are calculated on
the fly
"""
import pytest

import anndata
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import tempfile
from unittest.mock import patch

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
    'write_summary, cloud_safe, nodes_to_drop',
    itertools.product(
        [True, False],
        [True, False],
        [None, [('class', 'a'), ('subclass', 'subclass_5')]]))
def test_otf_smoke(
        tmp_dir_fixture,
        precomputed_path_fixture,
        raw_query_h5ad_fixture,
        write_summary,
        cloud_safe,
        nodes_to_drop):

    # record hash of precomputed stats file to make sure it
    # is not changed when nodes are dropped
    hasher = hashlib.md5()
    with open(precomputed_path_fixture, 'rb') as src:
        hasher.update(src.read())
    precompute_hash = hasher.hexdigest()

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
        'nodes_to_drop': nodes_to_drop
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

    hasher = hashlib.md5()
    with open(precomputed_path_fixture, 'rb') as src:
        hasher.update(src.read())
    final_hash = hasher.hexdigest()
    assert final_hash == precompute_hash


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
