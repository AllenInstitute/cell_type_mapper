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

from cell_type_mapper.utils.output_utils import (
    precomputed_stats_to_uns,
    uns_to_precomputed_stats
)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad,
    precompute_summary_stats_from_h5ad_and_lookup,
    precompute_summary_stats_from_h5ad_list_and_tree)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.utils.utils import (
    _clean_up,
    clean_for_json,
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

    # so that each processor has a different amount of
    # work to do in tests with n_processors > 1
    assert len(records) % 3 != 0

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


def create_h5ad(
        obs,
        x,
        tmp_dir,
        layer):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='h5ad_file_',
        suffix='.h5ad'
    )

    data = scipy_sparse.csr_matrix(x)

    if layer == 'X':
        xx = data
        layers = None
        raw = None
    elif layer == 'dummy':
        xx = np.zeros(shape(x), dtype=int)
        layers = {'dummy': data}
        raw = None
    elif layer == 'raw':
        xx = np.zeros(shape(x), dtype=int)
        layers = None
        raw = {'X': data}
    else:
        raise RuntimeError(
            "Test cannot parse layer '{layer}'"
        )

    a_data = anndata.AnnData(X=xx,
                             obs=obs,
                             layers=layers,
                             raw=raw)

    a_data.write_h5ad(h5ad_path)

    return h5ad_path


def create_many_h5ad(
        obs,
        x,
        tmp_dir,
        layer):
    """
    Store the data in multiple h5ad files;
    return a list to their paths
    """
    idx_arr =np.arange(x.shape[0])
    rng = np.random.default_rng(663344)
    rng.shuffle(idx_arr)
    n_per = len(idx_arr) // 4
    assert n_per > 2
    path_list = []
    for i0 in range(0, len(idx_arr), n_per):
       i1 = i0+n_per
       this_idx = idx_arr[i0:i1]
       this_obs = obs.iloc[this_idx]
       this_x = x[this_idx, :]
       csr = scipy_sparse.csr_matrix(this_x)
       if layer == 'X':
           xx = csr
           layers = None
           raw = None
       elif layer == 'dummy':
           xx = np.zeros(x.shape, dtype=int)
           layers = {'dummy': csr}
           raw = None
       elif layer == 'raw':
           xx = np.zeros(x.shape, dtype=int)
           layers = None
           raw = {'X': csr}
       else:
           raise RuntimeError(
               f"Test cannot parse layer '{layer}'"
           )

       this_a = anndata.AnnData(
           X=xx,
           layers=layers,
           raw=raw,
           obs=this_obs)

       this_path = mkstemp_clean(
           dir=tmp_dir,
           prefix='broken_up_h5ad',
           suffix='.h5ad')

       this_a.write_h5ad(this_path)
       path_list.append(this_path)
    return path_list


@pytest.fixture
def h5ad_input_path(
        request,
        tmp_dir_fixture,
        obs_fixture,
        raw_x_fixture,
        x_fixture):

    dataset = request.param['dataset']
    layer = request.param['layer']
    output_layer = layer

    if dataset == 'raw_data':
        return {'path': create_h5ad(
                            obs=obs_fixture,
                            x=raw_x_fixture,
                            tmp_dir=tmp_dir_fixture,
                            layer=layer),
                'normalization': 'raw',
                'layer': output_layer}
    elif dataset == 'log2_data':
        return {'path': create_h5ad(
                            obs=obs_fixture,
                            x=x_fixture,
                            tmp_dir=tmp_dir_fixture,
                            layer=layer),
                'normalization': 'log2CPM',
                'layer': output_layer}
    elif dataset == 'raw_data_list':
        return {'path': create_many_h5ad(
                            obs=obs_fixture,
                            x=raw_x_fixture,
                            tmp_dir=tmp_dir_fixture,
                            layer=layer),
                'normalization': 'raw',
                'layer': output_layer}
    elif dataset == 'log2_data_list':
        return {'path': create_many_h5ad(
                            obs=obs_fixture,
                            x=x_fixture,
                            tmp_dir=tmp_dir_fixture,
                            layer=layer),
                'normalization': 'log2CPM',
                'layer': output_layer}
    else:
        raise RuntimeError(
            f"Cannot parse request {request}"
        )

@pytest.mark.parametrize(
        'h5ad_input_path, n_processors',
        itertools.product(
            [{'dataset': 'raw_data',
              'layer': 'X'},
              {'dataset': 'log2_data',
               'layer': 'X'}],
            [1, 3]),
        indirect=['h5ad_input_path'])
def test_precompute_from_data(
        h5ad_input_path,
        records_fixture,
        baseline_stats_fixture,
        tmp_dir_fixture,
        n_processors):
    """
    Test the generation of precomputed stats file.

    The test checks results against known answers.
    """
    h5ad_path = h5ad_input_path['path']
    normalization = h5ad_input_path['normalization']
    layer = h5ad_input_path['layer']

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
        normalization=normalization,
        n_processors=n_processors,
        tmp_dir=tmp_dir_fixture)

    expected_tree = get_taxonomy_tree(
        obs_records=records_fixture,
        column_hierarchy=hierarchy)
    expected_tree = clean_for_json(expected_tree)
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
        'h5ad_input_path',
        [{'dataset': 'raw_data',
          'layer': 'X'},
          {'dataset': 'log2_data',
           'layer': 'X'}],
        indirect=['h5ad_input_path'])
def test_serialization_of_actual_precomputed_stats(
        h5ad_input_path,
        tmp_dir_fixture):
    """
    Test that serialization to uns works on an actual
    precomputed stats file
    """

    h5ad_path = h5ad_input_path['path']
    normalization = h5ad_input_path['normalization']
    layer = h5ad_input_path['layer']

    n_processors = 1

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture, prefix='stats'))

    hierarchy = ["level1", "level2", "class", "cluster"]

    stats_file = tmp_dir / "summary_stats_for_serialization.h5"
    assert not stats_file.is_file()

    precompute_summary_stats_from_h5ad(
        data_path=h5ad_path,
        column_hierarchy=hierarchy,
        taxonomy_tree=None,
        output_path=stats_file,
        rows_at_a_time=13,
        normalization=normalization,
        n_processors=n_processors,
        tmp_dir=tmp_dir_fixture)

    uns_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='h5ad_for_serialization_',
        suffix='.h5ad')

    # create a dummy h5ad file that we can serialize
    # the precomputed stats into
    rng = np.random.default_rng(88712)
    n_cells = 10
    n_genes = 22
    var = pd.DataFrame(
        [{'gene': f'g_{ii}'} for ii in range(n_genes)]
    ).set_index('gene')
    obs = pd.DataFrame(
        [{'cell': f'c_{ii}'} for ii in range(n_cells)]
    ).set_index('cell')
    a_data = anndata.AnnData(
        X=rng.random((n_cells, n_genes)),
        var=var,
        obs=obs,
        uns={'a': 'b', 'c': 'd'})
    a_data.write_h5ad(uns_path)

    # serialize and read back the precomputed stats to/from
    # the h5ad file's uns element
    uns_key = 'precomputed_stats'
    precomputed_stats_to_uns(
        precomputed_stats_path=stats_file,
        h5ad_path=uns_path,
        uns_key=uns_key)

    roundtrip_file = uns_to_precomputed_stats(
        h5ad_path=uns_path,
        uns_key=uns_key,
        tmp_dir=tmp_dir)

    # check that the two precomputed_stats files agree with
    # each other
    with h5py.File(stats_file, 'r') as expected_src:
        with h5py.File(roundtrip_file, 'r') as actual_src:
            assert set(expected_src.keys()) == set(actual_src.keys())
            for dataset_name in expected_src.keys():
                expected = expected_src[dataset_name][()]
                actual = actual_src[dataset_name][()]
                if isinstance(expected, np.ndarray):
                    np.testing.assert_allclose(
                        expected,
                        actual,
                        atol=0.0,
                        rtol=1.0e-6)
                else:
                    expected = json.loads(expected.decode('utf-8'))
                    actual = json.loads(actual.decode('utf-8'))
                    if expected != actual:
                        raise RuntimeError(
                            f"{dataset_name} does not match\n"
                            f"{expected}\n"
                            f"{actual}\n"
                        )


@pytest.mark.parametrize(
        'h5ad_input_path, omit_clusters, n_processors, copy_data_over',
        itertools.product(
            [{'dataset': 'raw_data_list',
              'layer': 'X'},
             {'dataset': 'log2_data_list',
              'layer': 'X'}],
            [True, False],
            [1, 3],
            [True, False]),
        indirect=['h5ad_input_path'])
def test_precompute_from_many_h5ad_with_lookup(
        records_fixture,
        obs_fixture,
        baseline_stats_fixture,
        tmp_dir_fixture,
        h5ad_input_path,
        omit_clusters,
        n_processors,
        copy_data_over):
    """
    Test the generation of precomputed stats file from many
    h5ad files at once.

    The test checks results against known answers.

    if omit_clusters, drop some clusters from the lookup table
    """
    path_list = h5ad_input_path['path']
    normalization = h5ad_input_path['normalization']
    layer = h5ad_input_path['layer']

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
        normalization=normalization,
        n_processors=n_processors,
        tmp_dir=tmp_dir_fixture,
        copy_data_over=copy_data_over)

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


@pytest.mark.parametrize('h5ad_input_path,use_cell_set,n_processors,copy_data_over',
        itertools.product(
            [{'dataset': 'raw_data_list',
              'layer': 'X'},
             {'dataset': 'log2_data_list',
              'layer': 'X'}],
            [True, False],
            [1, 3],
            [True, False]),
        indirect=['h5ad_input_path'])
def test_precompute_from_many_h5ad_with_tree(
        h5ad_input_path,
        records_fixture,
        obs_fixture,
        baseline_stats_fixture,
        tmp_dir_fixture,
        baseline_stats_fixture_limited_cells,
        cell_set_fixture,
        use_cell_set,
        n_processors,
        copy_data_over):
    """
    Test the generation of precomputed stats file from many
    h5ad files at once.

    The test checks results against known answers.
    """
    path_list = h5ad_input_path['path']
    normalization = h5ad_input_path['normalization']
    layer = h5ad_input_path['layer']

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
        cell_set=cell_set,
        n_processors=n_processors,
        tmp_dir=tmp_dir_fixture,
        copy_data_over=copy_data_over)

    with h5py.File(stats_file, 'r') as in_file:
        actual_tree = json.loads(
            in_file['taxonomy_tree'][()].decode('utf-8'))
    for k in ('metadata', 'alias_mapping'):
        if k in actual_tree:
            actual_tree.pop(k)

    assert clean_for_json(expected_tree) == actual_tree

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
