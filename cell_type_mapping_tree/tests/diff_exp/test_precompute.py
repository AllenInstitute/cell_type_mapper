import pytest

import anndata
import h5py
import json
import numpy as np
import pandas as pd
import scipy.sparse as scipy_sparse
import pathlib

from hierarchical_mapping.taxonomy.utils import (
    get_taxonomy_tree)

from hierarchical_mapping.cell_by_gene.utils import (
    convert_to_cpm)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.utils.utils import (
    _clean_up,
    json_clean_dict)


@pytest.fixture
def ncols():
    return 71


@pytest.fixture
def l1_to_l2_fixture():
    """
    Fixture modeling which level 2 objects belong
    to level 1
    """
    forward = {"l1a": set(["l2a", "l2d", "l2e"]),
               "l1b": set(["l2b", "l2f"]),
               "l1c": set(["l2c"])}

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward


@pytest.fixture
def l2_to_class_fixture():
    """
    Fixture modeling which class objects belong
    to which level 2 objects
    """
    forward = {"l2a": set(["c4", "c5"]),
               "l2b": set(["c1", "c6"]),
               "l2c": set(["c3"]),
               "l2d": set(["c2", "c7", "c8"]),
               "l2e": set(["c9"]),
               "l2f": set(["c10", "c11"])}

    backward = dict()
    for k in forward:
        for i in forward[k]:
            backward[i] = k
    return forward, backward

@pytest.fixture
def class_to_cluster_fixture(l2_to_class_fixture):
    """
    Fixture modeling which cluster objects belong
    to which class objects
    """
    list_of_classes = list(l2_to_class_fixture[1].keys())

    forward = dict()
    backward = dict()
    ct = 0
    for c in list_of_classes:
        forward[c] = set()
        for ii in range(4):
            this = f"clu_{ct}"
            ct += 1
            backward[this] = c
            forward[c].add(this)

    return forward, backward


@pytest.fixture
def records_fixture(
         class_to_cluster_fixture,
         l2_to_class_fixture,
         l1_to_l2_fixture):
    rng = np.random.default_rng(871234)
    cluster_list = list(class_to_cluster_fixture[1].keys())
    records = []
    for ii in range(7):
        for clu in cluster_list:
            cl = class_to_cluster_fixture[1][clu]
            l2 = l2_to_class_fixture[1][cl]
            l1 = l1_to_l2_fixture[1][l2]
            this = {"cluster": clu,
                    "class": cl,
                    "level2": l2,
                    "level1": l1,
                    "garbage": rng.integers(8, 1000)}
            records.append(this)

    # so that not every cluster has the same number
    # of cells
    for ii in range(2*len(cluster_list)+len(cluster_list)//3):
        clu = rng.choice(cluster_list)
        cl = class_to_cluster_fixture[1][clu]
        l2 = l2_to_class_fixture[1][cl]
        l1 = l1_to_l2_fixture[1][l2]
        this = {"cluster": clu,
                "class": cl,
                "level2": l2,
                "level1": l1,
                "garbage": rng.integers(8, 1000)}
        records.append(this)


    rng.shuffle(records)
    return records

@pytest.fixture
def obs_fixture(records_fixture):
    return pd.DataFrame(records_fixture)


@pytest.fixture
def nrows(records_fixture):
    return len(records_fixture)

@pytest.fixture
def raw_x_fixture(ncols, nrows):
    rng = np.random.default_rng(66213)
    data = np.zeros(nrows*ncols, dtype=np.float64)
    chosen_dex = rng.choice(np.arange(nrows*ncols, dtype=int),
                            nrows*ncols//7,
                            replace=False)
    data[chosen_dex] = rng.random(len(chosen_dex))*10000.0
    data = data.reshape((nrows, ncols))
    # set one row to have a CPM = 1 gene
    data[5, : ] =0
    data[5, 16] = 999999
    data[5, 12] = 1
    return data


@pytest.fixture
def x_fixture(raw_x_fixture):
    cpm_data = convert_to_cpm(raw_x_fixture)
    return np.log2(cpm_data+1.0)


@pytest.fixture
def h5ad_path_fixture(
        obs_fixture,
        x_fixture,
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('anndata'))
    a_data = anndata.AnnData(X=scipy_sparse.csr_matrix(x_fixture),
                             obs=obs_fixture,
                             dtype=x_fixture.dtype)
    h5ad_path = tmp_dir / 'h5ad_file.h5ad'
    a_data.write_h5ad(h5ad_path)
    import h5py
    with h5py.File(h5ad_path, 'r', swmr=True) as in_file:
        d = in_file['X']['data']
    yield h5ad_path
    _clean_up(tmp_dir)


@pytest.fixture
def raw_h5ad_path_fixture(
        obs_fixture,
        raw_x_fixture,
        tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('anndata'))
    a_data = anndata.AnnData(X=scipy_sparse.csr_matrix(raw_x_fixture),
                             obs=obs_fixture,
                             dtype=raw_x_fixture.dtype)
    h5ad_path = tmp_dir / 'h5ad_file.h5ad'
    a_data.write_h5ad(h5ad_path)
    import h5py
    with h5py.File(h5ad_path, 'r', swmr=True) as in_file:
        d = in_file['X']['data']
    yield h5ad_path
    _clean_up(tmp_dir)


@pytest.fixture
def baseline_stats_fixture(
        records_fixture,
        x_fixture,
        ncols):

    results = dict()
    for i_row, record in enumerate(records_fixture):
        cluster = record["cluster"]
        if cluster not in results:
            results[cluster] = {
                "n_cells": 0,
                "sum": np.zeros(ncols, dtype=float),
                "sumsq": np.zeros(ncols, dtype=float),
                "gt0": np.zeros(ncols, dtype=int),
                "gt1": np.zeros(ncols, dtype=int),
                "ge1": np.zeros(ncols, dtype=int)}
        results[cluster]["n_cells"] += 1
        results[cluster]["sum"] += x_fixture[i_row, :]
        results[cluster]["sumsq"] += x_fixture[i_row, :]**2
        for i_col in range(ncols):
            if x_fixture[i_row, i_col] > 0:
                results[cluster]["gt0"][i_col] += 1
                if x_fixture[i_row, i_col] > 1:
                    results[cluster]["gt1"][i_col] += 1
                if x_fixture[i_row, i_col] >= 1:
                    results[cluster]["ge1"][i_col] += 1
    return results


@pytest.mark.parametrize(
        'use_raw',
        [True, False])
def test_precompute_from_data(
        h5ad_path_fixture,
        raw_h5ad_path_fixture,
        records_fixture,
        baseline_stats_fixture,
        tmp_path_factory,
        use_raw):
    """
    Test the generation of precomputed stats file.

    The test checks results against known answers.
    """

    if use_raw:
        h5ad_path = raw_h5ad_path_fixture
        normalization = 'raw'
    else:
        h5ad_path = h5ad_path_fixture
        normalization = 'log2CPM'

    tmp_dir = pathlib.Path(tmp_path_factory.mktemp("stats_from_contig"))

    hierarchy = ["level1", "level2", "class", "cluster"]

    stats_file = tmp_dir / "summary_stats.h5"
    assert not stats_file.is_file()

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path,
        column_hierarchy=hierarchy,
        taxonomy_tree=None,
        output_path=stats_file,
        rows_at_a_time=13,
        normalization=normalization)

    expected_tree = get_taxonomy_tree(
        obs_records=records_fixture,
        column_hierarchy=hierarchy)
    expected_tree = json_clean_dict(expected_tree)
    with h5py.File(stats_file, 'r') as in_file:
        actual_tree = json.loads(
            in_file['taxonomy_tree'][()].decode('utf-8'))
    for k in ('metadata', 'alias_mapping'):
        if k in actual_tree:
            actual_tree.pop(k)

    assert expected_tree == actual_tree

    assert stats_file.is_file()
    with h5py.File(stats_file, "r") as in_file:
        cluster_to_row = json.loads(
                    in_file["cluster_to_row"][()].decode("utf-8"))

        n_cells = in_file["n_cells"][()]
        sum_data = in_file["sum"][()]
        sumsq_data = in_file["sumsq"][()]
        gt0 = in_file["gt0"][()]
        gt1 = in_file["gt1"][()]
        ge1 = in_file["ge1"][()]

    assert not np.array_equal(gt1, ge1)
    assert len(cluster_to_row) == len(baseline_stats_fixture)
    for cluster in cluster_to_row:
        idx = cluster_to_row[cluster]

        np.testing.assert_allclose(
            sum_data[idx, :],
            baseline_stats_fixture[cluster]["sum"],
            rtol=1.0e-6)

        np.testing.assert_allclose(
            sumsq_data[idx, :],
            baseline_stats_fixture[cluster]["sumsq"],
            rtol=1.0e-6)

        np.testing.assert_array_equal(
            gt0[idx, :],
            baseline_stats_fixture[cluster]["gt0"])

        np.testing.assert_array_equal(
            gt1[idx, :],
            baseline_stats_fixture[cluster]["gt1"])

        np.testing.assert_array_equal(
            ge1[idx, :],
            baseline_stats_fixture[cluster]["ge1"])

        assert n_cells[idx] == baseline_stats_fixture[cluster]["n_cells"]

    _clean_up(tmp_dir)
