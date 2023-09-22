import pytest

import anndata
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import scipy.sparse as scipy_sparse
import pathlib
import tempfile

from cell_type_mapper.taxonomy.utils import (
    get_taxonomy_tree)

from cell_type_mapper.cell_by_gene.utils import (
    convert_to_cpm)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad,
    precompute_summary_stats_from_h5ad_and_lookup,
    precompute_summary_stats_from_h5ad_list_and_tree)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.utils.utils import (
    _clean_up,
    json_clean_dict,
    mkstemp_clean)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('precompute_'))
    yield tmp_dir
    _clean_up(tmp_dir)


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

    for ii, r in enumerate(records):
        r['cell_id'] = f'cell_{ii}'

    rng.shuffle(records)

    return records


@pytest.fixture
def obs_fixture(records_fixture):
    return pd.DataFrame(records_fixture).set_index('cell_id')


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
        tmp_dir_fixture):
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture, prefix='anndata_'))
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
        tmp_dir_fixture):
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture, prefix='raw_anndata'))
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


@pytest.fixture
def cell_set_fixture(records_fixture):
    cell_id_list = [r['cell_id'] for r in records_fixture]
    cell_id_list.sort()
    rng = np.random.default_rng(77123)
    return set(rng.choice(cell_id_list,
                          len(cell_id_list)//3,
                          replace=False))

@pytest.fixture
def baseline_stats_fixture_limited_cells(
        records_fixture,
        x_fixture,
        ncols,
        cell_set_fixture):
    """
    same as baseline_stats_fixture, but only using cells specified
    in cell_set_fixture
    """

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

        if record['cell_id'] not in cell_set_fixture:
            continue
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


@pytest.fixture
def many_h5ad_fixture(
        obs_fixture,
        x_fixture,
        tmp_dir_fixture):
    """
    Store the data in multiple h5ad files;
    return a list to their paths
    """
    idx_arr =np.arange(x_fixture.shape[0])
    rng = np.random.default_rng(663344)
    rng.shuffle(idx_arr)
    n_per = len(idx_arr) // 4
    assert n_per > 2
    path_list = []
    for i0 in range(0, len(idx_arr), n_per):
       i1 = i0+n_per
       this_idx = idx_arr[i0:i1]
       this_obs = obs_fixture.iloc[this_idx]
       this_x = x_fixture[this_idx, :]
       csr = scipy_sparse.csr_matrix(this_x)
       this_a = anndata.AnnData(X=csr, obs=this_obs, dtype=this_x.dtype)
       this_path = mkstemp_clean(
           dir=tmp_dir_fixture,
           prefix='broken_up_h5ad',
           suffix='.h5ad')
       this_a.write_h5ad(this_path)
       path_list.append(this_path)
    return path_list


@pytest.fixture
def many_raw_h5ad_fixture(
        obs_fixture,
        raw_x_fixture,
        tmp_dir_fixture):
    """
    Store the data in multiple h5ad files;
    return a list to their paths
    """
    idx_arr =np.arange(raw_x_fixture.shape[0])
    rng = np.random.default_rng(456312)
    rng.shuffle(idx_arr)
    n_per = len(idx_arr) // 4
    assert n_per > 2
    path_list = []
    for i0 in range(0, len(idx_arr), n_per):
       i1 = i0+n_per
       this_idx = idx_arr[i0:i1]
       this_obs = obs_fixture.iloc[this_idx]
       this_x = raw_x_fixture[this_idx, :]
       csr = scipy_sparse.csr_matrix(this_x)
       this_a = anndata.AnnData(X=csr, obs=this_obs, dtype=this_x.dtype)
       this_path = mkstemp_clean(
           dir=tmp_dir_fixture,
           prefix='broken_up_h5ad',
           suffix='.h5ad')
       this_a.write_h5ad(this_path)
       path_list.append(this_path)
    return path_list

@pytest.mark.parametrize(
        'use_raw',
        [True, False])
def test_precompute_from_data(
        h5ad_path_fixture,
        raw_h5ad_path_fixture,
        records_fixture,
        baseline_stats_fixture,
        tmp_dir_fixture,
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

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture, prefix='stats'))

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


@pytest.mark.parametrize(
        'use_raw, omit_clusters',
        itertools.product([True, False], [True, False]))
def test_precompute_from_many_h5ad_with_lookup(
        many_h5ad_fixture,
        many_raw_h5ad_fixture,
        records_fixture,
        obs_fixture,
        baseline_stats_fixture,
        tmp_dir_fixture,
        use_raw,
        omit_clusters):
    """
    Test the generation of precomputed stats file from many
    h5ad files at once.

    The test checks results against known answers.

    if omit_clusters, drop some clusters from the lookup table
    """
    if use_raw:
        path_list = many_raw_h5ad_fixture
        normalization = 'raw'
    else:
        path_list = many_h5ad_fixture
        normalization = 'log2CPM'

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture, prefix='stats'))

    hierarchy = ["level1", "level2", "class", "cluster"]

    stats_file = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='stats_from_many_',
            suffix='.h5'))

    expected_tree = get_taxonomy_tree(
        obs_records=records_fixture,
        column_hierarchy=hierarchy)

    if omit_clusters:
        key_list = list(expected_tree['cluster'].keys())
        to_omit = set([key_list[0], key_list[5]])
    else:
        to_omit = None

    cell_name_to_cluster_name = {
        str(cell): cluster
        for cell, cluster in zip(obs_fixture.index.values,
                                 obs_fixture['cluster'].values)
        if to_omit is None or cluster not in to_omit}

    cluster_to_output_row = {
        n:ii
        for ii, n in enumerate(expected_tree['cluster'].keys())}

    gene_names = list(read_df_from_h5ad(path_list[0], 'var').index.values)

    precompute_summary_stats_from_h5ad_and_lookup(
        data_path_list=path_list,
        cell_name_to_cluster_name=cell_name_to_cluster_name,
        cluster_to_output_row=cluster_to_output_row,
        output_path=stats_file,
        rows_at_a_time=13,
        normalization=normalization)

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

        if to_omit is not None:
            if cluster in to_omit:
                assert n_cells[idx] == 0
                np.testing.assert_allclose(
                        sum_data[idx, :],
                        np.zeros(len(gene_names)),
                        atol=0.0, rtol=1.0e-6)

                np.testing.assert_allclose(
                        sumsq_data[idx, :],
                        np.zeros(len(gene_names)),
                        atol=0.0, rtol=1.0e-6)

                np.testing.assert_allclose(
                        gt0[idx, :],
                        np.zeros(len(gene_names)),
                        atol=0.0, rtol=1.0e-6)

                np.testing.assert_allclose(
                        gt1[idx, :],
                        np.zeros(len(gene_names)),
                        atol=0.0, rtol=1.0e-6)

                np.testing.assert_allclose(
                        ge1[idx, :],
                        np.zeros(len(gene_names)),
                        atol=0.0, rtol=1.0e-6)
                continue

        assert n_cells[idx] == baseline_stats_fixture[cluster]["n_cells"]

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

    _clean_up(tmp_dir)


@pytest.mark.parametrize('use_raw,use_cell_set',
        itertools.product([True, False], [True, False]))
def test_precompute_from_many_h5ad_with_tree(
        many_h5ad_fixture,
        many_raw_h5ad_fixture,
        records_fixture,
        obs_fixture,
        baseline_stats_fixture,
        tmp_dir_fixture,
        baseline_stats_fixture_limited_cells,
        cell_set_fixture,
        use_raw,
        use_cell_set):
    """
    Test the generation of precomputed stats file from many
    h5ad files at once.

    The test checks results against known answers.
    """
    if use_raw:
        path_list = many_raw_h5ad_fixture
        normalization = 'raw'
    else:
        path_list = many_h5ad_fixture
        normalization = 'log2CPM'

    if use_cell_set:
        cell_set = cell_set_fixture
        ground_truth = baseline_stats_fixture_limited_cells
    else:
        cell_set = None
        ground_truth = baseline_stats_fixture

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture, prefix='stats'))

    hierarchy = ["level1", "level2", "class", "cluster"]

    stats_file = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='stats_from_many_',
            suffix='.h5'))

    expected_tree = get_taxonomy_tree(
        obs_records=records_fixture,
        column_hierarchy=hierarchy)

    # munge tree so that leaf node maps to cell name,
    # rather than rows in x_fixture
    expected_tree.pop('cluster')

    leaf_to_cell = dict()
    for record in records_fixture:
        cluster = record['cluster']
        cell = record['cell_id']
        if cluster not in leaf_to_cell:
            leaf_to_cell[cluster] = []
        leaf_to_cell[cluster].append(cell)
    expected_tree['cluster'] = leaf_to_cell

    taxonomy_tree = TaxonomyTree(data=expected_tree)

    precompute_summary_stats_from_h5ad_list_and_tree(
        data_path_list=path_list,
        taxonomy_tree=taxonomy_tree,
        output_path=stats_file,
        rows_at_a_time=13,
        normalization=normalization,
        cell_set=cell_set)

    with h5py.File(stats_file, 'r') as in_file:
        actual_tree = json.loads(
            in_file['taxonomy_tree'][()].decode('utf-8'))
    for k in ('metadata', 'alias_mapping'):
        if k in actual_tree:
            actual_tree.pop(k)

    assert json_clean_dict(expected_tree) == actual_tree

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
    assert len(cluster_to_row) == len(ground_truth)
    for cluster in cluster_to_row:
        idx = cluster_to_row[cluster]

        assert n_cells[idx] == ground_truth[cluster]["n_cells"]

        np.testing.assert_allclose(
            sum_data[idx, :],
            ground_truth[cluster]["sum"],
            rtol=1.0e-6)

        np.testing.assert_allclose(
            sumsq_data[idx, :],
            ground_truth[cluster]["sumsq"],
            rtol=1.0e-6)

        np.testing.assert_array_equal(
            gt0[idx, :],
            ground_truth[cluster]["gt0"])

        np.testing.assert_array_equal(
            gt1[idx, :],
            ground_truth[cluster]["gt1"])

        np.testing.assert_array_equal(
            ge1[idx, :],
            ground_truth[cluster]["ge1"])

    _clean_up(tmp_dir)
