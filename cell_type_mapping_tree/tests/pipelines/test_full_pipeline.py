import pytest

import h5py
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.utils.taxonomy_utils import (
    get_taxonomy_tree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('full_pipeline'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def taxonomy_tree_fixture(
        records_fixture,
        column_hierarchy):
    return get_taxonomy_tree(
        obs_records=records_fixture,
        column_hierarchy=column_hierarchy)


def test_all_of_it(
        tmp_dir_fixture,
        h5ad_path_fixture,
        column_hierarchy
        ):

    precomputed_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path_fixture,
        column_hierarchy=column_hierarchy,
        output_path=precomputed_path,
        rows_at_a_time=10)

    # make sure it is not empty
    with h5py.File(precomputed_path, 'r') as in_file:
        assert len(in_file.keys()) > 0
