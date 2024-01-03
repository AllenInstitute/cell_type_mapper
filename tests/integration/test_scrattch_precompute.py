import pytest

import anndata
import h5py
import itertools
import json
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.cell_by_gene.utils import (
    convert_to_cpm)

from cell_type_mapper.cli.precompute_stats_scrattch import (
    PrecomputationScrattchRunner)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('precompute_scrattch_')
    yield str(tmp_dir.resolve().absolute())
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def n_genes():
    return 112


@pytest.fixture(scope='module')
def n_cells():
    return 2000


@pytest.fixture(scope='module')
def raw_cell_by_gene_fixture(
        n_cells,
        n_genes):
    """
    Return a dict mapping cell names to gene expression
    profiles
    """
    rng = np.random.default_rng(671231)
    result = dict()
    for i_cell in range(n_cells):
        cell_name = f'cell_{i_cell}'
        expression = rng.integers(0, 15, n_genes)
        result[cell_name] = expression
    return result


@pytest.fixture(scope='module')
def taxonomy_data_fixture(raw_cell_by_gene_fixture):

    rng = np.random.default_rng(433221)

    hierarchy = ['class', 'subclass', 'cluster']
    n_clusters = 31
    n_subclasses = 7
    n_classes = 3
    cluster_list = [f'cluster_{ii}' for ii in range(n_clusters)]
    tree_data = dict()
    tree_data['class'] = {f'class_{ii}': []
                          for ii in range(n_classes)}
    tree_data['subclass'] = {f'subclass_{ii}': []
                             for ii in range(n_subclasses)}
    tree_data['cluster'] = {f'cluster_{ii}': []
                            for ii in range(n_clusters)}

    class_list = list(tree_data['class'].keys())
    class_list.sort()

    subclass_list = list(tree_data['subclass'].keys())
    subclass_list.sort()

    cluster_list = list(tree_data['cluster'].keys())
    cluster_list.sort()

    cluster_to_subclass = dict()
    subclass_to_class = dict()
    cell_to_cluster = dict()
    for subclass in subclass_list:
        this_class = rng.choice(class_list)
        subclass_to_class[subclass] = this_class
        tree_data['class'][this_class].append(subclass)
    for cluster in cluster_list:
        this_subclass = rng.choice(subclass_list)
        cluster_to_subclass[cluster] = this_subclass
        tree_data['subclass'][this_subclass].append(cluster)
    for cell in raw_cell_by_gene_fixture:
        this_cluster = rng.choice(cluster_list)
        cell_to_cluster[cell] = this_cluster
        tree_data['cluster'][this_cluster].append(cell)

    tree_data['hierarchy'] = hierarchy

    return {
        'tree_data': tree_data,
        'cell_to_cluster': cell_to_cluster,
        'cluster_to_subclass': cluster_to_subclass,
        'subclass_to_class': subclass_to_class
    }


@pytest.fixture(scope='module')
def log2_cell_by_gene_fixture(
        raw_cell_by_gene_fixture):
   result = dict()
   for cell in raw_cell_by_gene_fixture:
       cpm = convert_to_cpm(np.array([raw_cell_by_gene_fixture[cell]]))
       result[cell] = np.log2(1.0+cpm[0, :])
   return result        


@pytest.fixture(scope='module')
def taxonomy_fixture(taxonomy_data_fixture):
    return TaxonomyTree(data=taxonomy_data_fixture['tree_data'])


@pytest.fixture(scope='module')
def cluster_stats_fixture(
        log2_cell_by_gene_fixture,
        taxonomy_data_fixture,
        n_genes):
    result = dict()
    cell_to_cluster = taxonomy_data_fixture['cell_to_cluster']
    cluster_list = list(cell_to_cluster.values())
    cluster_list.sort()
    for cluster in cluster_list:
        result[cluster] = {'n_cells': 0,
                        'sum': np.zeros(n_genes),
                        'sumsq': np.zeros(n_genes),
                        'ge1': np.zeros(n_genes),
                        'gt0': np.zeros(n_genes),
                        'gt1': np.zeros(n_genes)}

    for cell in cell_to_cluster:
        cluster = cell_to_cluster[cell]
        result[cluster]['n_cells'] += 1
        expression = log2_cell_by_gene_fixture[cell]
        result[cluster]['sum'] += expression
        result[cluster]['sumsq'] += expression**2
        ge1 = (expression >= 1.0)
        result[cluster]['ge1'] += ge1
        gt0 = (expression > 0.0)
        result[cluster]['gt0'] += gt0
        gt1 = (expression > 1.0)
        result[cluster]['gt1'] += gt1

    return result


@pytest.fixture(scope='module')
def obs_fixture(taxonomy_data_fixture, taxonomy_fixture):
    cell_to_cluster = taxonomy_data_fixture['cell_to_cluster']
    cluster_to_subclass = taxonomy_data_fixture['cluster_to_subclass']
    subclass_to_class = taxonomy_data_fixture['subclass_to_class']

    data = []
    cell_list = list(cell_to_cluster.keys())
    cell_list.sort()
    for cell in cell_list:
        cluster = cell_to_cluster[cell]
        subclass = cluster_to_subclass[cluster]
        class_ = subclass_to_class[subclass]
        this = {
            'cell_id': cell,
            'cluster': cluster,
            'subclass': subclass,
            'class': class_}
        if class_ == 'class_0':
            n_c0 += 1
        data.append(this)
    obs = pd.DataFrame(data).set_index('cell_id')
    return obs


@pytest.fixture(scope='function')
def raw_h5ad_fixture(
        obs_fixture,
        raw_cell_by_gene_fixture,
        n_cells,
        n_genes,
        tmp_dir_fixture,
        request):

    x = np.zeros((n_cells, n_genes), dtype=int)
    for i_cell, cell in enumerate(obs_fixture.index.values):
        x[i_cell, :] = raw_cell_by_gene_fixture[cell]
    if request.param == 'csr':
        x = scipy.sparse.csr_matrix(x)
    elif request.param == 'csc':
        x = scipy.sparse.csc_matrix(x)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix=f'raw_scrattch_{request.param}_',
        suffix='.h5ad')

    var = pd.DataFrame(
        [{'gene': f'g_{ii}'}
         for ii in range(n_genes)]).set_index('gene')

    src = anndata.AnnData(
        X=x,
        obs=obs_fixture,
        var=var)

    src.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.fixture(scope='function')
def log2_h5ad_fixture(
        obs_fixture,
        log2_cell_by_gene_fixture,
        n_cells,
        n_genes,
        tmp_dir_fixture,
        request):

    x = np.zeros((n_cells, n_genes), dtype=float)
    for i_cell, cell in enumerate(obs_fixture.index.values):
        x[i_cell, :] = log2_cell_by_gene_fixture[cell]
    if request.param == 'csr':
        x = scipy.sparse.csr_matrix(x)
    elif request.param == 'csc':
        x = scipy.sparse.csc_matrix(x)

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix=f'raw_scrattch_{request.param}_',
        suffix='.h5ad')

    var = pd.DataFrame(
        [{'gene': f'g_{ii}'}
         for ii in range(n_genes)]).set_index('gene')

    src = anndata.AnnData(
        X=x,
        obs=obs_fixture,
        var=var)

    src.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.mark.parametrize(
        'raw_h5ad_fixture, log2_h5ad_fixture, n_processors',
        [('csc', 'csc', 1),
         ('csc', 'csc', 3),
         ('csr', 'csr', 1),
         ('csr', 'csr', 3),
         ('dense', 'dense', 1),
         ('dense', 'dense', 3)
        ],
        indirect=['raw_h5ad_fixture', 'log2_h5ad_fixture'])
def test_precompute_scrattch_cli(
        taxonomy_fixture,
        cluster_stats_fixture,
        n_processors,
        tmp_dir_fixture,
        raw_h5ad_fixture,
        log2_h5ad_fixture):

    for normalization in ('log2CPM', 'raw'):
        if normalization == 'raw':
            input_path = raw_h5ad_fixture
        elif normalization == 'log2CPM':
            input_path = log2_h5ad_fixture
        else:
            raise RuntimeError(
                f"Unsure about normalization {normalization}")

        output_path = mkstemp_clean(
            dir=tmp_dir_fixture,
            prefix='precompute_from_scrattch_',
            suffix='.h5')

        config = {
            'h5ad_path': input_path,
            'n_processors': n_processors,
            'normalization': normalization,
            'tmp_dir': tmp_dir_fixture,
            'output_path': output_path,
            'hierarchy': ['class', 'subclass', 'cluster']
        }

        runner = PrecomputationScrattchRunner(
            args=[],
            input_data=config)

        runner.run()

        actual_path = pathlib.Path(output_path)
        assert actual_path.is_file()

        src = anndata.read_h5ad(input_path, backed='r')
        gene_names = src.var.index.values

        with h5py.File(actual_path, 'r') as src:
            assert 'metadata' in src
            cluster_to_row = json.loads(
                src['cluster_to_row'][()].decode('utf-8'))
            assert len(cluster_to_row) == len(cluster_stats_fixture)
            np.testing.assert_array_equal(
                np.array(json.loads(src['col_names'][()].decode('utf-8'))),
                gene_names)
            for cluster in cluster_to_row:
                row_idx = cluster_to_row[cluster]
                expected = cluster_stats_fixture[cluster]
                assert src['n_cells'][row_idx] == expected['n_cells']
                for k in ('sum', 'sumsq'):
                    np.testing.assert_allclose(
                        src[k][row_idx, :],
                        expected[k],
                        atol=0.0,
                        rtol=1.0e-6)

                for k in ('gt1', 'gt0', 'ge1'):
                    np.testing.assert_array_equal(
                        src[k][row_idx, :],
                        expected[k])

        actual_taxonomy = TaxonomyTree.from_precomputed_stats(actual_path)
        assert actual_taxonomy.hierarchy == taxonomy_fixture.hierarchy
        for level in actual_taxonomy.hierarchy:
            actual_nodes = actual_taxonomy.nodes_at_level(level)
            for node in taxonomy_fixture.nodes_at_level(level):
                expected_children = taxonomy_fixture.children(level=level, node=node)
                if len(expected_children) == 0:
                    assert node not in actual_nodes
                else:
                   if level != actual_taxonomy.leaf_level:
                        # cells in actual_taxonomy are referred to by row number;
                        # in taxonomy_fixture they are referred to by cell_id
                        assert set(actual_taxonomy.children(level=level, node=node)) == set(expected_children)
