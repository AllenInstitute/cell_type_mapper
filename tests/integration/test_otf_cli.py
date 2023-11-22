"""
tests for the CLI tool that maps to markers which are calculated on
the fly
"""
import pytest

import anndata
import json
import tempfile
from unittest.mock import patch

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.cli.map_to_on_the_fly_markers import (
    OnTheFlyMapper)


@pytest.mark.parametrize(
    'write_summary', [True, False])
def test_otf_smoke(
        tmp_dir_fixture,
        precomputed_path_fixture,
        raw_query_h5ad_fixture,
        write_summary):

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
        'summary_metadata_path': metadata_path
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
