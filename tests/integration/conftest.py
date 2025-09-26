import pytest

import anndata
import copy
import h5py
import json
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)

from cell_type_mapper.diff_exp.markers import (
    find_markers_for_all_taxonomy_pairs)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_reference_markers)


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
    for i_sub, subclass in enumerate(subclasses):
        n_sub = rng.integers(2, 4)
        if i_sub == 0:
            n_sub = 7
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
def noisier_raw_reference_cell_x_gene(
        n_reference_cells,
        taxonomy_tree_dict,
        cluster_to_signal,
        reference_gene_names):
    """
    Reference data for cases where we don't need to assume what the
    result of mapping/marker finding is, but need to test that the
    result changes due to configuration parameter changes (increase
    the noise amplitude relative to default raw_reference_cell_x_gene)
    """
    rng = np.random.default_rng(671231)
    n_genes = len(reference_gene_names)
    x_data = np.zeros((n_reference_cells, n_genes),
                      dtype=float)
    for cl in taxonomy_tree_dict['cluster']:
        signal_lookup = cluster_to_signal[cl]
        for i_cell in taxonomy_tree_dict['cluster'][cl]:
            noise = rng.random(n_genes)
            noise_amp = 0.5*rng.random()
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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
def noisier_raw_reference_h5ad_fixture(
        noisier_raw_reference_cell_x_gene,
        reference_gene_names,
        obs_records_fixture,
        tmp_dir_fixture):

    var_data = [{'gene_name': g, 'garbage': ii}
                for ii, g in enumerate(reference_gene_names)]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    obs = pd.DataFrame(obs_records_fixture)
    obs = obs.set_index('cell_id')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(
            X=noisier_raw_reference_cell_x_gene,
            obs=obs,
            var=var,
            dtype=noisier_raw_reference_cell_x_gene.dtype)

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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


@pytest.fixture(scope='module')
def raw_query_h5ad_alt_gene_col_fixture(
        raw_query_h5ad_fixture,
        tmp_dir_fixture):
    """
    The same data as in raw_query_h5ad_fixture,
    but the gene identifiers are in a different column
    in var (not the index).
    """
    src = anndata.read_h5ad(
        raw_query_h5ad_fixture).to_memory()
    src.file.close()
    new_var = src.var.reset_index()
    new_var = new_var.set_index('garbage')

    new_a = anndata.AnnData(
        obs=src.obs,
        X=src.X,
        var=new_var
    )
    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='alt_gene_col_',
        suffix='.h5ad'
    )
    new_a.write_h5ad(dst_path)
    return dst_path


@pytest.fixture(scope='module')
def precomputed_path_fixture(
        tmp_dir_fixture,
        raw_reference_h5ad_fixture,
        taxonomy_tree_dict):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

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

    return precomputed_path


@pytest.fixture(scope='module')
def noisier_precomputed_path_fixture(
        tmp_dir_fixture,
        noisier_raw_reference_h5ad_fixture,
        taxonomy_tree_dict):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(
            data=taxonomy_tree_dict)

    precomputed_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_',
        suffix='.h5')

    precompute_summary_stats_from_h5ad(
        data_path=noisier_raw_reference_h5ad_fixture,
        column_hierarchy=None,
        taxonomy_tree=taxonomy_tree,
        output_path=precomputed_path,
        rows_at_a_time=1000,
        normalization='raw')

    # make sure it is not empty
    with h5py.File(precomputed_path, 'r') as in_file:
        assert len(in_file.keys()) > 0

    return precomputed_path


@pytest.fixture(scope='module')
def ref_marker_path_fixture(
        tmp_dir_fixture,
        precomputed_path_fixture,
        taxonomy_tree_dict):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(
            data=taxonomy_tree_dict)

    ref_marker_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='reference_markers_',
        suffix='.h5')

    find_markers_for_all_taxonomy_pairs(
        precomputed_stats_path=precomputed_path_fixture,
        taxonomy_tree=taxonomy_tree,
        output_path=ref_marker_path,
        tmp_dir=tmp_dir_fixture,
        max_gb=2)

    metadata = {
        'precomputed_path': str(precomputed_path_fixture)
    }

    with h5py.File(ref_marker_path, 'a') as dst:
        dst.create_dataset(
            'metadata',
            data=json.dumps(metadata).encode('utf-8'))

    with h5py.File(ref_marker_path, 'r') as in_file:
        assert len(in_file.keys()) > 0
        assert in_file['sparse_by_pair/up_gene_idx'].shape[0] > 0
        assert in_file['sparse_by_pair/down_gene_idx'].shape[0] > 0
        assert in_file['sparse_by_gene/up_pair_idx'].shape[0] > 0
        assert in_file['sparse_by_gene/down_pair_idx'].shape[0] > 0

    return ref_marker_path


@pytest.fixture(scope='module')
def marker_cache_path_fixture(
        tmp_dir_fixture,
        taxonomy_tree_dict,
        ref_marker_path_fixture,
        query_gene_names):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(
            data=taxonomy_tree_dict)

    marker_cache_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='ref_and_query_markers_',
        suffix='.h5')

    create_marker_cache_from_reference_markers(
        output_cache_path=marker_cache_path,
        input_cache_path=ref_marker_path_fixture,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        n_per_utility=7,
        n_processors=3,
        behemoth_cutoff=1000000)

    with h5py.File(marker_cache_path, 'r') as in_file:
        assert len(in_file['None']['reference'][()]) > 0

    return marker_cache_path


@pytest.fixture(scope='module')
def full_marker_name_fixture(
        marker_cache_path_fixture,
        taxonomy_tree_dict):
    """
    Return a list of the names of all of the genes that
    were found as query markers
    """
    gene_name_list = []
    with h5py.File(marker_cache_path_fixture) as src:
        reference_gene_names = json.loads(
            src['reference_gene_names'][()].decode('utf-8'))
        gene_name_list += [
            reference_gene_names[idx]
            for idx in src['None']['reference'][()]]
        for level in taxonomy_tree_dict['hierarchy'][:-1]:
            for node in taxonomy_tree_dict[level]:
                gene_name_list += [
                    reference_gene_names[idx]
                    for idx in src[level][node]['reference'][()]]
    gene_name_list = list(set(gene_name_list))
    gene_name_list.sort()
    return gene_name_list


@pytest.fixture(scope='module')
def noisy_raw_reference_h5ad_fixture(
        obs_records_fixture,
        reference_gene_names,
        tmp_dir_fixture):
    rng = np.random.default_rng(223123)
    n_cells = len(obs_records_fixture)
    n_genes = len(reference_gene_names)
    data = np.zeros(n_cells*n_genes, dtype=int)
    chosen = rng.choice(np.arange(len(data)), len(data)//3, replace=False)
    data[chosen] = rng.integers(100, 1000)
    data = data.reshape((n_cells, n_genes))

    var_data = [{'gene_name': g, 'garbage': ii}
                for ii, g in enumerate(reference_gene_names)]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    obs = pd.DataFrame(obs_records_fixture)
    obs = obs.set_index('cell_id')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(
            X=scipy.sparse.csr_matrix(data),
            obs=obs,
            var=var,
            dtype=int)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='noisy_reference_',
            suffix='.h5ad'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope='module')
def noisy_precomputed_stats_fixture(
        tmp_dir_fixture,
        taxonomy_tree_dict,
        noisy_raw_reference_h5ad_fixture):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(data=taxonomy_tree_dict)

        output_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                prefix='noisy_precomputed_stats_',
                suffix='.h5'))
        precompute_summary_stats_from_h5ad(
            data_path=noisy_raw_reference_h5ad_fixture,
            column_hierarchy=None,
            taxonomy_tree=taxonomy_tree,
            output_path=output_path,
            rows_at_a_time=10000,
            normalization='raw')

    return output_path


@pytest.fixture(scope='module')
def noisy_raw_query_h5ad_fixture(
        query_gene_names,
        tmp_dir_fixture):

    rng = np.random.default_rng(77665544)

    n_cells = 500
    n_genes = len(query_gene_names)

    data = rng.integers(100, 110, (n_cells, n_genes))

    var_data = [
        {'gene_name': g, 'garbage': ii}
        for ii, g in enumerate(query_gene_names)
    ]

    var = pd.DataFrame(var_data)
    var = var.set_index('gene_name')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(
            X=data,
            var=var,
            uns={'AIBS_CDM_gene_mapping': {'a': 'b', 'c': 'd'}},
            dtype=int)

    h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir_fixture,
            suffix='.h5ad'))
    a_data.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope='module')
def noisy_marker_gene_lookup_fixture(
        tmp_dir_fixture,
        reference_gene_names,
        taxonomy_tree_dict):

    output_path = pathlib.Path(
        mkstemp_clean(
           dir=tmp_dir_fixture,
           prefix='marker_lookup_',
           suffix='.json'))

    rng = np.random.default_rng(77123)
    markers = dict()
    markers['None'] = list(rng.choice(reference_gene_names, 27, replace=False))
    for level in taxonomy_tree_dict['hierarchy'][:-1]:
        for node in taxonomy_tree_dict[level]:
            if len(taxonomy_tree_dict[level][node]) == 1:
                continue
            node_key = f"{level}/{node}"
            markers[node_key] = list(
                rng.choice(
                    reference_gene_names,
                    rng.integers(11, 34),
                    replace=False))

    with open(output_path, 'w') as dst:
        dst.write(json.dumps(markers))

    return output_path
