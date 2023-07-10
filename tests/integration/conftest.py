import pytest

import anndata
import copy
import numpy as np
import pandas as pd
import pathlib

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('known_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def n_reference_cells():
    return 10000


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def reference_gene_names(gene_names):
    return gene_names[0]

@pytest.fixture(scope='module')
def query_gene_names(gene_names):
    return gene_names[1]

@pytest.fixture(scope='module')
def marker_gene_names(gene_names):
    return gene_names[2]

@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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
        var=var,
        dtype=raw_reference_cell_x_gene.dtype)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path

@pytest.fixture(scope='module')
def expected_cluster_fixture(
        taxonomy_tree_dict):
    n_query_cells = 555
    cluster_list = list(taxonomy_tree_dict['cluster'].keys())
    rng = np.random.default_rng(87123)
    chosen_clusters = rng.choice(
        cluster_list,
        n_query_cells,
        replace=True)
    return chosen_clusters


@pytest.fixture(scope='module')
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

@pytest.fixture(scope='module')
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
        var=var,
        uns={'AIBS_CDM_gene_mapping': {'a': 'b', 'c': 'd'}},
        dtype=raw_query_cell_x_gene_fixture.dtype)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path


