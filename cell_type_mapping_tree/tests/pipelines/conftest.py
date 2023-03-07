# this file will define a "raw" dataset for
# pipeline tests (as much as possible)

import pytest

import anndata
import numpy as np
import os
import pandas as pd
import pathlib
import scipy.sparse as scipy_sparse
import tempfile

from hierarchical_mapping.utils.utils import (
    _clean_up)


@pytest.fixture
def column_hierarchy():
    return ["level1", "level2", "class", "cluster"]


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
def n_genes(
        class_to_cluster_fixture,
        l2_to_class_fixture,
        l1_to_l2_fixture):
    cluster_to_class = class_to_cluster_fixture[1]
    class_to_l2 = l2_to_class_fixture[1]
    l2_to_l1 = l1_to_l2_fixture[1]

    ct = len(cluster_to_class)
    ct += len(class_to_l2)
    ct += len(l2_to_l1)
    ct += len(l1_to_l2_fixture[0])
    return ct


@pytest.fixture
def gene_names(n_genes):
    return [f"gene_{ii}" for ii in range(n_genes)]

    
@pytest.fixture
def cluster_list(class_to_cluster_fixture):
    return list(class_to_cluster_fixture[1].keys())


@pytest.fixture
def cluster_to_signal(
        cluster_list,
        class_to_cluster_fixture,
        l2_to_class_fixture,
        l1_to_l2_fixture,
        n_genes):
    cluster_to_class = class_to_cluster_fixture[1]
    class_to_l2 = l2_to_class_fixture[1]
    l2_to_l1 = l1_to_l2_fixture[1]

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
        result[cluster] = signal
    return result

@pytest.fixture
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


@pytest.fixture
def n_cells(records_fixture):
    return len(records_fixture)

@pytest.fixture
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
        pure_noise = np.where(signal<1.0e-6)[0]
        chosen = rng.choice(pure_noise,
                            len(pure_noise)//3,
                            replace=False)
        noise[chosen] = 0.0

        signal += noise

        data[idx, :] = signal

    return data


@pytest.fixture
def h5ad_path_fixture(
        cell_x_gene_fixture,
        records_fixture,
        gene_names,
        tmp_path_factory):

    tmp_dir = pathlib.Path(
                tmp_path_factory.mktemp('pipeline_anndata'))

    a_data_path = tmp_dir / 'test_cell_x_gene.h5ad'

    obs = pd.DataFrame(records_fixture)

    var_data = [{'gene_name': g}
                for g in gene_names]

    var = pd.DataFrame(var_data).set_index('gene_name')

    csr = scipy_sparse.csr_matrix(cell_x_gene_fixture)

    a_data = anndata.AnnData(X=csr,
                             obs=obs,
                             var=var,
                             dtype=csr.dtype)
    a_data.write_h5ad(a_data_path)

    yield a_data_path

    _clean_up(tmp_dir)


@pytest.fixture
def query_h5ad_path_fixture(
        gene_names,
        tmp_path_factory,
        l1_to_l2_fixture):
    """
    Returns a path to an H5ad file that can be queried
    against the clusters in h5ad_fixture_path
    """
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('query_data'))
    h5ad_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5ad')
    os.close(h5ad_path[0])
    h5ad_path = pathlib.Path(h5ad_path[1])

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

    n_query_genes = len(query_genes)
    n_query_cells = 5555

    x_data = np.zeros(n_query_genes*n_query_cells, dtype=float)
    chosen_dex = rng.choice(np.arange(n_query_cells*n_query_genes),
                            2*n_query_cells*n_query_genes//3,
                            replace=False)
    x_data[chosen_dex] = rng.random(len(chosen_dex))
    x_data = x_data.reshape((n_query_cells, n_query_genes))
    x_data = scipy_sparse.csr_matrix(x_data)

    var_data = [{'gene_name': g}
                for g in query_genes]
    var = pd.DataFrame(var_data).set_index('gene_name')

    a_data = anndata.AnnData(
                    X=x_data,
                    var=var,
                    dtype=x_data.dtype)
    a_data.write_h5ad(h5ad_path)
    del a_data

    yield h5ad_path

    _clean_up(tmp_dir)
