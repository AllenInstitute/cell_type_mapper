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

from hierarchical_mapping.diff_exp.diff_exp import (
    diffexp_score,
    score_all_taxonomy_pairs)

from hierarchical_mapping.zarr_creation.zarr_from_h5ad import (
    contiguous_zarr_from_h5ad)

from hierarchical_mapping.diff_exp.precompute import (
    precompute_summary_stats_from_contiguous_zarr)


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
        tmp_path_factory):

    tmp_dir = pathlib.Path(
                tmp_path_factory.mktemp('pipeline_anndata'))

    a_data_path = tmp_dir / 'test_cell_x_gene.h5ad'

    obs = pd.DataFrame(records_fixture)

    csr = scipy_sparse.csr_matrix(cell_x_gene_fixture)

    a_data = anndata.AnnData(X=csr, obs=obs)
    a_data.write_h5ad(a_data_path)

    yield a_data_path

    _clean_up(tmp_dir)


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


def test_pipeline(
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

    #spock
    #need to compare to brute force outputs

    _clean_up(tmp_dir)
