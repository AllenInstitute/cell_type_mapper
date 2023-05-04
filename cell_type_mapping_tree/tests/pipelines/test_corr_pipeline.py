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

from hierarchical_mapping.taxonomy.utils import (
    get_taxonomy_tree,
    _get_rows_from_tree)

from hierarchical_mapping.corr.correlate_cells import (
    correlate_cells)

from hierarchical_mapping.zarr_creation.zarr_from_h5ad import (
    contiguous_zarr_from_h5ad)

from hierarchical_mapping.diff_exp.precompute import (
    precompute_summary_stats_from_contiguous_zarr)


def test_correlation_pipeline_smoketest(
        h5ad_path_fixture,
        query_h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory):

    tmp_dir = pathlib.Path(
            tmp_path_factory.mktemp('corr_pipeline'))
    zarr_path = tmp_dir / 'as_zarr.zarr'
    precompute_path = tmp_dir / 'precomputed.h5'
    corr_path = tmp_dir / 'corr.h5'

    contiguous_zarr_from_h5ad(
        h5ad_path=h5ad_path_fixture,
        zarr_path=zarr_path,
        taxonomy_hierarchy=column_hierarchy,
        zarr_chunks=100000,
        n_processors=3)

    assert not precompute_path.is_file()

    precompute_summary_stats_from_contiguous_zarr(
        zarr_path=zarr_path,
        output_path=precompute_path,
        rows_at_a_time=1000,
        n_processors=3)

    assert precompute_path.is_file()
    assert not corr_path.is_file()

    correlate_cells(
        query_path=query_h5ad_path_fixture,
        precomputed_path=precompute_path,
        output_path=corr_path,
        rows_at_a_time=1000,
        n_processors=3)

    assert corr_path.is_file()

    # make sure correlation matrix is not empty
    with h5py.File(corr_path, 'r') as in_file:
        corr = in_file['correlation'][()]
    zeros = np.zeros(corr.shape, dtype=corr.dtype)
    assert not np.allclose(zeros, corr)

    _clean_up(tmp_dir)
