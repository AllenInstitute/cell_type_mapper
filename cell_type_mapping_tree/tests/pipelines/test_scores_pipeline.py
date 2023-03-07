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
    _get_rows_from_tree)

from hierarchical_mapping.diff_exp.scores import (
    diffexp_score,
    score_all_taxonomy_pairs)

from hierarchical_mapping.zarr_creation.zarr_from_h5ad import (
    contiguous_zarr_from_h5ad)

from hierarchical_mapping.diff_exp.precompute import (
    precompute_summary_stats_from_contiguous_zarr)


@pytest.fixture
def tree_fixture(
        records_fixture,
        column_hierarchy):
    return get_taxonomy_tree(
                obs_records=records_fixture,
                column_hierarchy=column_hierarchy)


@pytest.fixture
def brute_force_de_scores(
        cell_x_gene_fixture,
        tree_fixture):
    data = cell_x_gene_fixture
    result = dict()
    hierarchy = tree_fixture['hierarchy']
    for level in hierarchy:
        this_level = dict()
        node_list = list(tree_fixture[level].keys())
        node_list.sort()
        for i1 in range(len(node_list)):
            node1 = node_list[i1]
            row1 = _get_rows_from_tree(
                        tree=tree_fixture,
                        level=level,
                        this_node=node1)
            row1 = np.sort(np.array(row1))
            mu1 = np.mean(data[row1, :], axis=0)
            var1 = np.var(data[row1, :], axis=0, ddof=1)
            n1 = len(row1)
            this_level[node1] = dict()
            for i2 in range(i1+1, len(node_list), 1):
                node2 = node_list[i2]
                row2 = _get_rows_from_tree(
                            tree=tree_fixture,
                            level=level,
                            this_node=node2)

                row2 = np.sort(np.array(row2))
                mu2 = np.mean(data[row2, :], axis=0)
                var2 = np.var(data[row2,:], axis=0, ddof=1)
                n2 = len(row2)
                scores = diffexp_score(
                            mean1=mu1,
                            var1=var1,
                            n1=n1,
                            mean2=mu2,
                            var2=var2,
                            n2=n2)
                this_level[node1][node2] = scores
        result[level] = this_level
    return result


def test_scoring_pipeline(
        h5ad_path_fixture,
        brute_force_de_scores,
        column_hierarchy,
        tmp_path_factory):

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('pipeline_process'))
    zarr_path = tmp_dir / 'zarr.zarr'
    hdf5_tmp = tmp_dir / 'hdf5'
    hdf5_tmp.mkdir()

    contiguous_zarr_from_h5ad(
        h5ad_path=h5ad_path_fixture,
        zarr_path=zarr_path,
        taxonomy_hierarchy=column_hierarchy,
        zarr_chunks=100000,
        n_processors=3)

    precompute_path = tmp_dir / 'precomputed.h5'
    assert not precompute_path.is_file()

    precompute_summary_stats_from_contiguous_zarr(
        zarr_path=zarr_path,
        output_path=precompute_path,
        rows_at_a_time=1000,
        n_processors=3)

    assert precompute_path.is_file()

    metadata = json.load(
            open(zarr_path / 'metadata.json', 'rb'))
    taxonomy_tree = metadata["taxonomy_tree"]

    actual_de = score_all_taxonomy_pairs(
            precomputed_stats_path=precompute_path,
            taxonomy_tree=taxonomy_tree,
            gt1_threshold=0,
            gt0_threshold=1)

    assert len(actual_de) == len(brute_force_de_scores)
    for level in brute_force_de_scores:
        actual_level = actual_de[level]
        expected_level = brute_force_de_scores[level]
        assert len(actual_level) == len(expected_level)
        for node1 in expected_level:
            actual_node1 = actual_level[node1]
            expected_node1 = expected_level[node1]
            assert len(actual_node1) == len(expected_node1)
            for node2 in expected_node1:
                np.testing.assert_allclose(
                    actual_node1[node2]['score'],
                    expected_node1[node2],
                    atol=1.0e-5,
                    rtol=1.0e-4)

    _clean_up(tmp_dir)
