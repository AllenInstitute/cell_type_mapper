import pytest

import anndata
import h5py
import numpy as np
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up,
    json_clean_dict)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.taxonomy.utils import (
    get_taxonomy_tree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers)

from cell_type_mapper.type_assignment.election_runner import (
    run_type_assignment_on_h5ad)


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
        taxonomy_tree=None,
        output_path=precomputed_path,
        rows_at_a_time=10)

    # make sure it is not empty
    with h5py.File(precomputed_path, 'r') as in_file:
        assert len(in_file.keys()) > 0

    ref_marker_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_markers_',
        suffix='.h5')

    taxonomy_tree = TaxonomyTree(data=taxonomy_tree_fixture)

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precomputed_path,
        taxonomy_tree=taxonomy_tree,
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

    create_marker_cache_from_reference_markers(
        output_cache_path=marker_cache_path,
        input_cache_path=ref_marker_path,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=1000000)

    with h5py.File(marker_cache_path, 'r') as in_file:
        assert len(in_file['None']['reference'][()]) > 0

    result = run_type_assignment_on_h5ad(
        query_h5ad_path=query_h5ad_path_fixture,
        precomputed_stats_path=precomputed_path,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree,
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
