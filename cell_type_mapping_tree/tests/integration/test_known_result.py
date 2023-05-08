"""
Run the full pipeline, testing a case where we know what
clusters cells should be assigned to
"""

import pytest

import anndata
import copy
import h5py
import json
import numpy as np
import pandas as pd
import pathlib
import tempfile

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from hierarchical_mapping.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from hierarchical_mapping.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from hierarchical_mapping.type_assignment.marker_cache_v2 import (
    create_marker_gene_cache_v2)

from hierarchical_mapping.type_assignment.election import (
    run_type_assignment_on_h5ad)

from hierarchical_mapping.cli.hierarchical_mapping import (
    run_mapping)


@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('known_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def n_reference_cells():
    return 100000

@pytest.fixture
def taxonomy_tree_dict(n_reference_cells):
    rng = np.random.default_rng(223112)
    hierarchy = ['class', 'subclass', 'cluster']
    tree = dict()
    tree['hierarchy'] = hierarchy
    subclasses = []
    ct = 0
    tree['class'] = dict()
    for class_name in ('a', 'b', 'c'):
        n_sub = rng.integers(2, 4)
        tree['class'][class_name] = []
        for ii in range(n_sub):
            name = f"subclass_{ct}"
            ct += 1
            subclasses.append(name)
            tree['class'][class_name].append(name)

    clusters = []
    tree['subclass'] = dict()
    for subclass in subclasses:
        n_sub = rng.integers(2, 4)
        tree['subclass'][subclass] = []
        for ii in range(n_sub):
            name = f"cl{ct}"
            ct += 1
            tree['subclass'][subclass].append(name)
            clusters.append(name)

    tree['cluster'] = dict()
    for ii in range(n_reference_cells):
        chosen = rng.choice(clusters, 1)[0]
        if chosen not in tree['cluster']:
            tree['cluster'][chosen] = []
        tree['cluster'][chosen].append(ii)

    for cluster in clusters:
        assert len(tree['cluster'][cluster]) > 3

    return tree

@pytest.fixture
def obs_records_fixture(taxonomy_tree_dict):
    cluster_to_subclass = dict()
    for subclass in taxonomy_tree_dict['subclass']:
        for cl in taxonomy_tree_dict['subclass'][subclass]:
            cluster_to_subclass[cl] = subclass
    subclass_to_class = dict()
    for class_name in taxonomy_tree_dict['class']:
        for subclass in taxonomy_tree_dict['class'][class_name]:
            subclass_to_class[subclass] = class_name

    row_to_cluster = dict()
    for cluster in taxonomy_tree_dict['cluster']:
        for idx in taxonomy_tree_dict['cluster'][cluster]:
            row_to_cluster[idx] = cluster
    cell_id_list = list(row_to_cluster.keys())
    cell_id_list.sort()
    results = []
    for idx in cell_id_list:
        cluster = row_to_cluster[idx]
        subclass = cluster_to_subclass[cluster]
        class_name = subclass_to_class[subclass]
        results.append(
            {'cell_id': idx,
             'cluster': cluster,
             'subclass': subclass,
             'class': class_name})
    return results


@pytest.fixture
def gene_names(
        taxonomy_tree_dict):
    n_clusters = len(taxonomy_tree_dict['cluster'])
    n_reference_genes = 12*n_clusters
    reference_gene_names = [f"gene_{ii}"
                            for ii in range(n_reference_genes)]
    marker_genes = copy.deepcopy(reference_gene_names)

    query_gene_names = copy.deepcopy(reference_gene_names)
    for ii in range(12):
        query_gene_names.append(f"query_nonsense_{ii}")

    rng = np.random.default_rng(11723)
    rng.shuffle(query_gene_names)

    # append some nonsense genes (genes that should not be markers)
    for ii in range(n_clusters):
        reference_gene_names.append(f"nonsense_{ii}")
    rng.shuffle(reference_gene_names)

    return reference_gene_names, query_gene_names, marker_genes


@pytest.fixture
def reference_gene_names(gene_names):
    return gene_names[0]

@pytest.fixture
def query_gene_names(gene_names):
    return gene_names[1]

@pytest.fixture
def marker_gene_names(gene_names):
    return gene_names[2]


@pytest.fixture
def cluster_to_signal(
        taxonomy_tree_dict,
        marker_gene_names):

    result = dict()
    rng = np.random.default_rng(66713)
    for ii, cl in enumerate(taxonomy_tree_dict['cluster']):
        genes = marker_gene_names[ii*7:(ii+1)*7]
        assert len(genes) == 7
        signal = np.power(8, rng.integers(2, 7, len(genes)))
        result[cl] = {n: s for n, s, in zip(genes, signal)}
    return result


@pytest.fixture
def raw_reference_cell_x_gene(
        n_reference_cells,
        taxonomy_tree_dict,
        cluster_to_signal,
        reference_gene_names):
    rng = np.random.default_rng(22312)
    n_genes = len(reference_gene_names)
    x_data = np.zeros((n_reference_cells, n_genes),
                      dtype=float)
    for cl in taxonomy_tree_dict['cluster']:
        signal_lookup = cluster_to_signal[cl]
        for i_cell in taxonomy_tree_dict['cluster'][cl]:
            noise = rng.random(n_genes)
            noise_amp = 0.1*rng.random()
            signal_amp = (2.0+rng.random())
            data = noise_amp*noise
            for i_gene, g in enumerate(reference_gene_names):
                if g in signal_lookup:
                    data[i_gene] += signal_amp*signal_lookup[g]
            x_data[i_cell, :] = data
    return x_data


@pytest.fixture
def raw_reference_h5ad_fixture(
        raw_reference_cell_x_gene,
        reference_gene_names,
        obs_records_fixture,
        tmp_dir_fixture):

    var_data = [{'gene_name': g, 'garbage': ii}
                for ii, g in enumerate(reference_gene_names)]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    obs = pd.DataFrame(obs_records_fixture)
    obs = obs.set_index('cell_id')

    a_data = anndata.AnnData(
        X=raw_reference_cell_x_gene,
        obs=obs,
        var=var)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path

@pytest.fixture
def expected_cluster_fixture(
        taxonomy_tree_dict):
    n_query_cells = 5555
    cluster_list = list(taxonomy_tree_dict['cluster'].keys())
    rng = np.random.default_rng(87123)
    chosen_clusters = rng.choice(
        cluster_list,
        n_query_cells,
        replace=True)
    return chosen_clusters

@pytest.fixture
def raw_query_cell_x_gene_fixture(
        cluster_to_signal,
        expected_cluster_fixture,
        query_gene_names):
    n_cells = len(expected_cluster_fixture)
    n_genes = len(query_gene_names)
    x_data = np.zeros((n_cells, n_genes), dtype=float)
    rng = np.random.default_rng(665533)
    for i_cell in range(n_cells):
        cl = expected_cluster_fixture[i_cell]
        signal_lookup = cluster_to_signal[cl]
        noise = 0.1*rng.random(n_genes)
        noise_amp = rng.random()
        signal_amp = (2.0+rng.random())
        data = noise_amp*noise
        for i_gene, g in enumerate(query_gene_names):
            if g in signal_lookup:
                data[i_gene] += signal_amp*signal_lookup[g]
        x_data[i_cell, :] = data

    return x_data

@pytest.fixture
def raw_query_h5ad_fixture(
        raw_query_cell_x_gene_fixture,
        query_gene_names,
        tmp_dir_fixture):
    var_data = [
        {'gene_name': g, 'garbage': ii}
         for ii, g in enumerate(query_gene_names)]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    a_data = anndata.AnnData(
        X=raw_query_cell_x_gene_fixture,
        var=var)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path


def test_raw_pipeline(
        raw_reference_h5ad_fixture,
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        query_gene_names,
        tmp_dir_fixture):

    taxonomy_tree = TaxonomyTree(
        data=taxonomy_tree_dict)

    precomputed_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    precompute_summary_stats_from_h5ad(
        data_path=raw_reference_h5ad_fixture,
        column_hierarchy=None,
        taxonomy_tree=taxonomy_tree,
        output_path=precomputed_path,
        rows_at_a_time=1000,
        normalization='raw')

    # make sure it is not empty
    with h5py.File(precomputed_path, 'r') as in_file:
        assert len(in_file.keys()) > 0

    ref_marker_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_markers_',
        suffix='.h5')

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precomputed_path,
        taxonomy_tree=taxonomy_tree,
        output_path=ref_marker_path,
        tmp_dir=tmp_dir_fixture,
        max_bytes=6*1024)

    with h5py.File(ref_marker_path, 'r') as in_file:
        assert len(in_file.keys()) > 0
        assert in_file['up_regulated/data'][()].sum() > 0
        assert in_file['markers/data'][()].sum() > 0

    marker_cache_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='ref_and_query_markers_',
        suffix='.h5')

    create_marker_gene_cache_v2(
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
        query_h5ad_path=raw_query_h5ad_fixture,
        precomputed_stats_path=precomputed_path,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree,
        n_processors=3,
        chunk_size=100,
        bootstrap_factor=6.0/7.0,
        bootstrap_iteration=100,
        rng=np.random.default_rng(123545),
        normalization='raw')

    assert len(result) == len(expected_cluster_fixture)
    for cell in result:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]


@pytest.mark.parametrize('use_tree', [True, False])
def test_cli_pipeline(
        raw_reference_h5ad_fixture,
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        query_gene_names,
        tmp_dir_fixture,
        use_tree):

    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture)

    to_store = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture))

    precompute_out = to_store / 'precomputed.h5'
    ref_marker_out = to_store / 'ref_markers.h5'

    config = dict()
    config['tmp_dir'] = tmp_dir
    config['query_path'] = str(
        raw_query_h5ad_fixture.resolve().absolute())

    config['precomputed_stats'] = {
        'reference_path': str(raw_reference_h5ad_fixture.resolve().absolute()),
        'path': str(precompute_out),
        'normalization': 'raw'}

    if use_tree:
        tree_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.json')
        with open(tree_path, 'w') as out_file:
            out_file.write(json.dumps(taxonomy_tree_dict))
        config['precomputed_stats']['taxonomy_tree'] = tree_path
    else:
        config['precomputed_stats']['column_hierarchy'] = taxonomy_tree_dict['hierarchy']

    config['reference_markers'] = {
        'n_processors': 3,
        'max_bytes': 6*1024**2,
        'path': str(ref_marker_out)}

    config["query_markers"] = {
        'n_per_utility': 5,
        'n_processors': 3}

    config["type_assignment"] = {
        'n_processors': 3,
        'bootstrap_factor': 0.9,
        'bootstrap_iteration': 27,
        'rng_seed': 66234,
        'chunk_size': 1000,
        'normalization': 'raw'}

    assert not precompute_out.is_file()
    assert not ref_marker_out.is_file()

    log_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    output_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    run_mapping(
        config,
        output_path=output_path,
        log_path=log_path)

    assert precompute_out.is_file()
    assert ref_marker_out.is_file()

    # check for existence of marker summary
    with h5py.File(ref_marker_out, 'r') as in_file:
        for k in ('sparse/up_gene_idx',
                  'sparse/up_pair_idx',
                  'sparse/down_gene_idx',
                  'sparse/down_pair_idx'):
            assert k in in_file
            assert len(in_file[k][()]) > 0

    log = json.load(open(log_path, 'rb'))
    assert isinstance(log, list)
    assert len(log) > 0

    results = json.load(open(output_path, 'rb'))
    other_log = results["log"]

    # this is convoluted because the logger as
    # implemented prepends some timing information
    # to the log messages
    for msg in ("creating precomputed stats",
                "creating reference marker file"):
        for this_log in (log, other_log):
            found_it = False
            for line in this_log:
                if msg in line:
                    found_it = True
                    break
            assert found_it

    assert len(results["results"]) == len(expected_cluster_fixture)

    for cell in results["results"]:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]

    # ======== now run it, reusing the precomputed files =========
    config.pop('precomputed_stats')
    config.pop('reference_markers')
    precompute_str = str(precompute_out.resolve().absolute())
    ref_marker_str = str(ref_marker_out.resolve().absolute())

    config['precomputed_stats'] = {'path': precompute_str}
    config['reference_markers'] = {'path': ref_marker_str}

    log_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    output_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    run_mapping(
        config,
        output_path=output_path,
        log_path=log_path)

    log = json.load(open(log_path, 'rb'))
    assert isinstance(log, list)
    assert len(log) > 0

    results = json.load(open(output_path, 'rb'))
    other_log = results["log"]

    # make sure we did not create new stats/marker files
    # when we did not have to
    for msg in ("creating precomputed stats",
                "creating reference marker file"):
        for this_log in (log, other_log):
            found_it = False
            for line in this_log:
                if msg in line:
                    found_it = True
                    break
            assert not found_it

    assert len(results["results"]) == len(expected_cluster_fixture)

    for cell in results["results"]:
        cell_id = int(cell['cell_id'])
        actual_cluster = cell['cluster']['assignment']
        expected_cluster = expected_cluster_fixture[cell_id]
        assert actual_cluster == expected_cluster
        actual_sub = cell['subclass']['assignment']
        assert actual_cluster in taxonomy_tree_dict['subclass'][actual_sub]
        actual_class = cell['class']['assignment']
        assert actual_sub in taxonomy_tree_dict['class'][actual_class]


def test_cli_error_log(
        raw_reference_h5ad_fixture,
        raw_query_h5ad_fixture,
        expected_cluster_fixture,
        taxonomy_tree_dict,
        query_gene_names,
        tmp_dir_fixture):
    """
    Same as test_cli_pipeline except configured to fail so that we can
    check the log and make sure the error was captured
    """
    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir_fixture)

    to_store = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir_fixture))

    ref_marker_out = to_store / 'ref_markers.h5'


    # this will be a bad path
    precompute_out = '/nonexsistent/directory/precomputed.h5'

    config = dict()
    config['tmp_dir'] = tmp_dir
    config['query_path'] = str(
        raw_query_h5ad_fixture.resolve().absolute())

    config['precomputed_stats'] = {
        'reference_path': str(raw_reference_h5ad_fixture.resolve().absolute()),
        'path': precompute_out,
        'normalization': 'raw'}

    tree_path = mkstemp_clean(dir=tmp_dir_fixture, suffix='.json')
    with open(tree_path, 'w') as out_file:
            out_file.write(json.dumps(taxonomy_tree_dict))
    config['precomputed_stats']['taxonomy_tree'] = tree_path

    config['reference_markers'] = {
        'n_processors': 3,
        'max_bytes': 6*1024**2,
        'path': str(ref_marker_out)}

    config["query_markers"] = {
        'n_per_utility': 5,
        'n_processors': 3}

    config["type_assignment"] = {
        'n_processors': 3,
        'bootstrap_factor': 0.9,
        'bootstrap_iteration': 27,
        'rng_seed': 66234,
        'chunk_size': 1000,
        'normalization': 'raw'}

    log_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    output_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.json'))

    with pytest.raises(RuntimeError):
        run_mapping(
            config,
            output_path=output_path,
            log_path=log_path)

    log = json.load(open(log_path, 'rb'))

    found_error = False
    found_clean = False
    for line in log:
        if 'an ERROR occurred ====' in line:
            found_error = True
        if 'CLEANING UP' in line:
            found_clean = True
    assert found_error
    assert found_clean
