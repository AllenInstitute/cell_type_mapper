import pytest

import anndata
import h5py
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up,
    json_clean_dict)

from hierarchical_mapping.utils.taxonomy_utils import (
    get_taxonomy_tree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    create_marker_gene_cache_v2)

from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad)


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
    tree = get_taxonomy_tree(
        obs_records=records_fixture,
        column_hierarchy=column_hierarchy)
    tree = json_clean_dict(tree)
    return tree


def test_all_of_it(
        tmp_dir_fixture,
        h5ad_path_fixture,
        column_hierarchy,
        taxonomy_tree_fixture,
        query_h5ad_path_fixture
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

    ref_marker_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_markers_',
        suffix='.h5')

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precomputed_path,
        taxonomy_tree=taxonomy_tree_fixture,
        output_path=ref_marker_path,
        tmp_dir=tmp_dir_fixture,
        max_bytes=6*1024**2)

    with h5py.File(ref_marker_path, 'r') as in_file:
        assert len(in_file.keys()) > 0
        assert in_file['up_regulated/data'][()].sum() > 0
        assert in_file['markers/data'][()].sum() > 0

    q_data = anndata.read_h5ad(query_h5ad_path_fixture, backed='r')
    query_gene_names = list(q_data.var_names)

    marker_cache_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='ref_and_query_markers_',
        suffix='.h5')

    create_marker_gene_cache_v2(
        output_cache_path=marker_cache_path,
        input_cache_path=ref_marker_path,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree_fixture,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=1000000)

    with h5py.File(marker_cache_path, 'r') as in_file:
        assert len(in_file['None']['reference'][()]) > 0

    result = run_type_assignment_on_h5ad(
        query_h5ad_path=query_h5ad_path_fixture,
        precomputed_stats_path=precomputed_path,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree_fixture,
        n_processors=3,
        chunk_size=100,
        bootstrap_factor=0.9,
        bootstrap_iteration=100,
        rng=np.random.default_rng(123545))

    assert len(result) == q_data.X.shape[0]
    cell_id_set = set()
    for cell in result:
        cell_id_set.add(cell['cell_id'])
        for level in column_hierarchy:
            assert level in cell
            assert cell[level]['assignment'] is not None
    assert cell_id_set == set(q_data.obs_names)
