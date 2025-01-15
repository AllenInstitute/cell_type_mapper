import numpy as np
import h5py
import pathlib

from cell_type_mapper.utils.utils import (
    _clean_up)

from cell_type_mapper.corr.correlate_cells import (
    correlate_cells)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)


def test_correlation_pipeline_smoketest(
        h5ad_path_fixture,
        query_h5ad_path_fixture,
        column_hierarchy,
        tmp_path_factory):

    tmp_dir = pathlib.Path(
            tmp_path_factory.mktemp('corr_pipeline'))
    precompute_path = tmp_dir / 'precomputed.h5'
    corr_path = tmp_dir / 'corr.h5'

    assert not precompute_path.is_file()

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        taxonomy_tree=None,
        output_path=precompute_path,
        rows_at_a_time=1000)

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
