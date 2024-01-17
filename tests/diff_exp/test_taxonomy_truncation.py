import pytest

import anndata
import numpy as np
import pandas as pd
import scipy.sparse

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.diff_exp.precompute_from_anndata import (
    precompute_summary_stats_from_h5ad)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('truncation_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def cluster_to_parent_fixture():
    subclass_to_class = {
        f'subc_{ii}': f'class_{ii//3}'
        for ii in range(5)
    }

    supertype_to_subclass = {
        'supt_0': 'subc_0',
        'supt_1': 'subc_1',
        'supt_2': 'subc_1',
        'supt_3': 'subc_1',
        'supt_4': 'subc_2',
        'supt_5': 'subc_3',
        'supt_6': 'subc_3',
        'supt_7': 'subc_4'
    }

    cluster_to_supertype = {
        'clus_0': 'supt_0',
        'clus_1': 'supt_0',
        'clus_2': 'supt_1',
        'clus_3': 'supt_1',
        'clus_4': 'supt_2',
        'clus_5': 'supt_3',
        'clus_6': 'supt_4',
        'clus_7': 'supt_4',
        'clus_8': 'supt_4',
        'clus_9': 'supt_5',
        'clus_10': 'supt_6',
        'clus_11': 'supt_6',
        'clus_11': 'supt_6',
        'clus_12': 'supt_7',
        'clus_13': 'supt_7'
    }

    result = dict()
    for cluster in cluster_to_subclass:
        supertype = cluster_to_supertype[cluster]
        subclass = supertype_to_subclass[cluster]
        class_ = subclass_to_class[subclass]
        this = {'supertype': supertype,
                'subclass': subclass,
                'class': class_}
        result[cluster] = this

    return result


@pytest.fixture(scope='module')
def x_fixture():
    rng = np.random.default_rng(22887733)
    n_cells = 2000
    n_genes = 100
    n_tot = n_cells*n_genes
    data = np.zeros(n_tot, dtype=int)
    chosen = rng.choice(np.arange(n_tot), n_tot//3, replace=False)
    data[chosen] = rng.integers(1, 255, dtype=int)
    return data


@pytest.fixture(scope='module')
def h5ad_fixture(
        x_fixture,
        cluster_to_parent_fixture,
        tmp_dir_fixture):

    h5ad_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='scrattch_for_truncation_',
        suffix='.h5ad')

    var = pd.DataFrame(
        [{'gene_id': f'g_{ii}'}
         for ii in range(x_fixture.shape[0])]).set_index('gene_id')

    rng = np.random.default_rng(7611211)
    cluster_list = list(cluster_to_parent_fixture.keys())
    cell_records = []
    for i_cell in range(x_fixture.shape[0]):
        this = {'cell_id': f'c_{i_cell}'}
        if i_cell < 100:
            cluster = f'clus_{ii%14}'
        else:
            cluster = rng.choice(cluster_list)
        parents = cluster_to_parent_fixture[cluster]
        supertype = parents['supertype']
        subclass = parents['subclass']
        class_ = parents['class']
        this['cluster'] = cluster
        this['subclass'] = subclass
        this['supertype'] = supertype
        this['class'] = class_
        cell_records.append(this)

    obs = pd.DataFrame(cell_records).set_index('cell_id')

    x = scipy.sparse.csr_matrix(x_fixture)

    a_data = anndata.AnnData(
        X=x,
        obs=obs,
        var=var)

    a_data.write_h5ad(h5ad_path)

    return h5ad_path
