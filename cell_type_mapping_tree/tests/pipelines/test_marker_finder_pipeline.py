import pytest

import pandas as pd
import numpy as np
import h5py
import anndata
import pathlib
import json
import scipy.sparse as scipy_sparse

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.taxonomy_utils import (
    get_taxonomy_tree,
    _get_rows_from_tree,
    get_all_pairs)

from hierarchical_mapping.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from hierarchical_mapping.zarr_creation.zarr_from_h5ad import (
    contiguous_zarr_from_h5ad)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)


@pytest.fixture
def tree_fixture(
        records_fixture,
        column_hierarchy):
    return get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)



def test_marker_finding_pipeline(
        h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory,
        gene_names,
        tree_fixture):

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('pipeline_process'))
    hdf5_tmp = tmp_dir / 'hdf5'
    hdf5_tmp.mkdir()
    marker_path = tmp_dir / 'marker_results.h5'

    precompute_path = tmp_dir / 'precomputed.h5'
    assert not precompute_path.is_file()

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        output_path=precompute_path,
        rows_at_a_time=1000)

    assert precompute_path.is_file()

    taxonomy_tree = tree_fixture


    # make sure flush_every is not an integer
    # divisor of the number of sibling pairs
    flush_every = 11
    n_processors = 3
    siblings = get_all_pairs(tree_fixture)
    assert len(siblings) > (n_processors*flush_every)
    assert len(siblings) % (n_processors*flush_every) != 0

    assert not marker_path.is_file()
    find_markers_for_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            output_path=marker_path,
            n_processors=n_processors,
            tmp_dir=tmp_dir)

    assert marker_path.is_file()
    with h5py.File(marker_path, 'r') as in_file:
        assert 'markers/data' in in_file
        assert 'up_regulated/data' in in_file
        assert 'gene_names' in in_file
        assert 'pair_to_idx' in in_file

    _clean_up(tmp_dir)
