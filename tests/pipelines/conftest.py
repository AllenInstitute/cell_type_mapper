# this file will define a "raw" dataset for
# pipeline tests (as much as possible)

import pytest

import anndata
import numpy as np
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import warnings

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('pipeline_data'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def column_hierarchy():
    return ["level1", "level2", "class", "cluster"]


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def n_non_markers():
    """
    number of genes that will be pure noise
    (i.e. will never occur as marker genes)
    """
    return 10


@pytest.fixture(scope='module')
def n_genes(
        class_to_cluster_fixture,
        l2_to_class_fixture,
        l1_to_l2_fixture,
        n_non_markers):
    cluster_to_class = class_to_cluster_fixture[1]
    class_to_l2 = l2_to_class_fixture[1]
    l2_to_l1 = l1_to_l2_fixture[1]

    ct = len(cluster_to_class)
    ct += len(class_to_l2)
    ct += len(l2_to_l1)
    ct += len(l1_to_l2_fixture[0])

    # add some more genes that are not markers
    ct += n_non_markers
    return ct


@pytest.fixture(scope='module')
def gene_names(n_genes):
    return [f"gene_{ii}" for ii in range(n_genes)]


@pytest.fixture(scope='module')
def cluster_list(class_to_cluster_fixture):
    return list(class_to_cluster_fixture[1].keys())


@pytest.fixture(scope='module')
def cluster_to_signal(
        cluster_list,
        class_to_cluster_fixture,
        l2_to_class_fixture,
        l1_to_l2_fixture,
        n_genes,
        n_non_markers):
    cluster_to_class = class_to_cluster_fixture[1]
    class_to_l2 = l2_to_class_fixture[1]
    l2_to_l1 = l1_to_l2_fixture[1]

    # shuffle the genes so that marker vs. non-marker
    # isn't influenced by the order of the genes
    rng = np.random.default_rng(123)
    shuffle_order = np.arange(n_genes, dtype=int)
    rng.shuffle(shuffle_order)

    taxon_to_idx = dict()
    ct = 0
    for k in cluster_to_class:
        taxon_to_idx[k] = ct
        ct += 1
    for k in class_to_l2:
        taxon_to_idx[k] = ct
        ct += 1
    for k in l2_to_l1:
        taxon_to_idx[k] = ct
        ct += 1
    for k in l1_to_l2_fixture[0]:
        taxon_to_idx[k] = ct
        ct += 1

    result = dict()
    for cluster in cluster_to_class:
        cl = cluster_to_class[cluster]
        l2 = class_to_l2[cl]
        l1 = l2_to_l1[l2]
        signal = np.zeros(n_genes, dtype=float)
        for k in (cluster, cl, l2, l1):
            idx = taxon_to_idx[k]
            signal[idx] += 1.0
        signal[-n_non_markers:] = 0.0
        signal = signal[shuffle_order]
        result[cluster] = signal
    return result


@pytest.fixture(scope='module')
def records_fixture(
         class_to_cluster_fixture,
         l2_to_class_fixture,
         l1_to_l2_fixture,
         cluster_list):
    rng = np.random.default_rng(4433772)
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

    # so that not every leaf node has the same number
    # of cells in it
    n_clusters = len(cluster_list)
    for ii in range(2*n_clusters + n_clusters//3):
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


@pytest.fixture(scope='module')
def n_cells(records_fixture):
    return len(records_fixture)


@pytest.fixture(scope='module')
def cell_x_gene_fixture(
        n_genes,
        n_cells,
        cluster_to_signal,
        records_fixture):

    rng = np.random.default_rng(662233)

    data = np.zeros((n_cells, n_genes), dtype=float)

    for idx, record in enumerate(records_fixture):
        amp = (1.0+rng.random(n_genes))

        # baseline signal for a cell of this type
        signal = amp*cluster_to_signal[record["cluster"]]

        # generate random noise
        noise = 0.1*rng.random(n_genes)

        # zero out some of the noise so that
        # there are zero entries in the cell_x_gene
        # matrix
        pure_noise = np.where(signal < 1.0e-6)[0]
        chosen = rng.choice(pure_noise,
                            len(pure_noise)//3,
                            replace=False)
        noise[chosen] = 0.0

        signal += noise

        data[idx, :] = signal

    return data


@pytest.fixture(scope='module')
def h5ad_path_fixture(
        cell_x_gene_fixture,
        records_fixture,
        gene_names,
        tmp_dir_fixture):

    tmp_dir = tmp_dir_fixture

    a_data_path = mkstemp_clean(dir=tmp_dir, suffix='.h5ad')

    obs = pd.DataFrame(records_fixture)

    var_data = [{'gene_name': g}
                for g in gene_names]

    var = pd.DataFrame(var_data).set_index('gene_name')

    csr = scipy_sparse.csr_matrix(cell_x_gene_fixture)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(
            X=csr,
            obs=obs,
            var=var,
            dtype=csr.dtype)

    a_data.write_h5ad(a_data_path)

    return a_data_path


@pytest.fixture(scope='module')
def query_genes_fixture(
        gene_names,
        l1_to_l2_fixture):

    rng = np.random.default_rng(99887766)
    tot_genes = len(gene_names)
    query_genes = []
    for ii in range(tot_genes, tot_genes-len(l1_to_l2_fixture[0]), -1):
        query_genes.append(f"gene_{ii}")
    others = np.arange(0, tot_genes-len(l1_to_l2_fixture[0]))
    chosen = rng.choice(others, len(others)//4, replace=False)
    for ii in chosen:
        query_genes.append(f"gene_{ii}")
    for ii in range(14):
        query_genes.append(f"nonsense_{ii}")
    query_genes = list(set(query_genes))
    query_genes.sort()
    rng.shuffle(query_genes)
    return query_genes


@pytest.fixture(scope='module')
def query_log2_cell_x_gene_fixture(
        query_genes_fixture):
    rng = np.random.default_rng(76213)
    n_query_cells = 5555
    n_query_genes = len(query_genes_fixture)
    x_data = np.zeros(n_query_genes*n_query_cells, dtype=float)
    chosen_dex = rng.choice(np.arange(n_query_cells*n_query_genes),
                            2*n_query_cells*n_query_genes//3,
                            replace=False)
    x_data[chosen_dex] = rng.random(len(chosen_dex))
    x_data = x_data.reshape((n_query_cells, n_query_genes))
    return x_data


@pytest.fixture(scope='module')
def query_h5ad_path_fixture(
        query_genes_fixture,
        query_log2_cell_x_gene_fixture,
        tmp_dir_fixture,
        l1_to_l2_fixture):
    """
    Returns a path to an H5ad file that can be queried
    against the clusters in h5ad_fixture_path
    """
    tmp_dir = tmp_dir_fixture
    h5ad_path = pathlib.Path(
        mkstemp_clean(dir=tmp_dir, suffix='.h5ad'))

    query_genes = query_genes_fixture

    x_data = query_log2_cell_x_gene_fixture
    x_data = scipy_sparse.csr_matrix(x_data)

    var_data = [{'gene_name': g}
                for g in query_genes]
    var = pd.DataFrame(var_data).set_index('gene_name')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        a_data = anndata.AnnData(
                        X=x_data,
                        var=var,
                        dtype=x_data.dtype)

    a_data.write_h5ad(h5ad_path)
    del a_data

    return h5ad_path
