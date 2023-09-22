import pytest

import h5py
import itertools
import json
import numpy as np

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.scores import (
    diffexp_p_values)

from cell_type_mapper.utils.stats_utils import (
    boring_t_from_p_value)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('reference_markers_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def taxonomy_tree_fixture():

    data = {
        'hierarchy': ['class', 'cluster'],
        'class': {
            'A': ['a', 'c', 'd'],
            'B': ['b', 'e', 'f', 'g']
        },
        'cluster': {
            'a': [],
            'b': [],
            'c': [],
            'd': [],
            'e': [],
            'f': [],
            'g': []
        }
    }
    return TaxonomyTree(data=data)


@pytest.fixture(scope='module')
def n_genes():
    return 85

@pytest.fixture(scope='module')
def cluster_profile_fixture(taxonomy_tree_fixture, n_genes):
    """
    Dict mapping cluster name to 'sum', 'sumsq', 'ge1', 'n_cells'
    stats
    """
    cluster_names = taxonomy_tree_fixture.nodes_at_level(
        taxonomy_tree_fixture.leaf_level)

    rng = np.random.default_rng(881231)

    result = dict()
    for cluster in cluster_names:
        n_cells = rng.integers(100, 1000)
        data = rng.normal(0.0, 1.0, (n_cells, n_genes))
        data *= 10.0*rng.random(n_genes)
        data += 5.0*rng.random(n_genes)
        this_sum = data.sum(axis=0)
        this_sumsq = (data**2).sum(axis=0)
        this = {
            'sum': this_sum,
            'sumsq': this_sumsq,
            'n_cells': n_cells,
            'ge1': rng.integers(10, n_cells, n_genes)
        }
        result[cluster] = this
    return result


@pytest.fixture(scope='module')
def p_value_fixture(
        cluster_profile_fixture):
    """
    Dict mapping cluster pair to p-values by gene
    """
    cluster_names = list(cluster_profile_fixture.keys())
    result = dict()
    n_interesting = 0
    for cl0, cl1 in itertools.combinations(cluster_names, 2):
        if cl0 not in result:
            result[cl0] = dict()
        if cl1 not in result:
            result[cl1] = dict()
        data0 = cluster_profile_fixture[cl0]
        data1 = cluster_profile_fixture[cl1]

        mu0 = data0['sum']/data0['n_cells']
        var0 = (data0['sumsq']-data0['sum']**2/data0['n_cells'])/(data0['n_cells']-1)
        mu1 = data1['sum']/data1['n_cells']
        var1 = (data1['sumsq']-data1['sum']**2/data1['n_cells'])/(data1['n_cells']-1)

        p_values = diffexp_p_values(
            mean1=mu0,
            var1=var0,
            n1=data0['n_cells'],
            mean2=mu1,
            var2=var1,
            n2=data1['n_cells'],
            boring_t=None,
            big_nu=None,
            p_th=None)

        result[cl0][cl1] = p_values
        result[cl1][cl0] = p_values
        n_interesting += (p_values < 0.4).sum()
    assert n_interesting > 1000
    return result

@pytest.fixture(scope='module')
def penetrance_fixture(
       cluster_profile_fixture):
    """
    Dict mapping cluster pairs to penetrance stats
    (q1, qdiff, log2_fold)
    """
    result = dict()
    cluster_names = list(cluster_profile_fixture.keys())
    for cl0, cl1 in itertools.combinations(cluster_names, 2):
        if cl0 not in result:
            result[cl0] = dict()
        if cl1 not in result:
            result[cl1] = dict()
        data0 = cluster_profile_fixture[cl0]
        data1 = cluster_profile_fixture[cl1]

        mu0 = data0['sum']/data0['n_cells']
        mu1 = data1['sum']/data1['n_cells']

        pij0 = data0['ge1']/data0['n_cells']
        pij1 = data1['ge1']/data1['n_cells']

        q1_score = np.where(pij0>pij1, pij0, pij1)
        qdiff_score = np.abs(pij0-pij1)/q1_score
        log2f = np.abs(mu0-mu1)
        this = {'q1': q1_score, 'qdiff': qdiff_score, 'log2_fold': log2f}
        result[cl0][cl1] = this
        result[cl1][cl0] = this
    return result

@pytest.fixture(scope='module')
def threshold_mask_generator_fixture(
        penetrance_fixture,
        p_value_fixture,
        taxonomy_tree_fixture):
    """
    A dict mapping cluster pairs and threshold configs
    to unique validity masks

    Also, a list of configs applied to all cluster pairs
    and the validity masks expected from them

    *expected masks assume that n_valid is so large that only
    the min thresholds for penetrance stats are interesting
    """
    cluster_name_list = taxonomy_tree_fixture.nodes_at_level('cluster')

    result = dict()
    n_degenerate = 0
    n_tot = 0
    non_degenerate_configs = []
    for cl0, cl1 in itertools.combinations(cluster_name_list, 2):
        if cl0 not in result:
            result[cl0] = dict()
        if cl1 not in result:
            result[cl1] = dict()
        p_truth = p_value_fixture[cl0][cl1]
        penetrance_truth = penetrance_fixture[cl0][cl1]
        q1_truth = penetrance_truth['q1']
        qdiff_truth = penetrance_truth['qdiff']
        log2_truth = penetrance_truth['log2_fold']

        q1_min = np.quantile(q1_truth, 0.1)
        valid = (q1_truth > q1_min)
        assert valid.sum() < len(valid)
        n0 = valid.sum()
        qdiff_min = np.quantile(qdiff_truth[valid], 0.15)
        valid = np.logical_and(
            q1_truth > q1_min, qdiff_truth > qdiff_min)
        assert valid.sum() < n0
        n0 = valid.sum()
        assert (qdiff_truth > qdiff_min).sum() > n0
        log2_min = np.quantile(log2_truth[valid], 0.2)
        valid = np.logical_and(valid, log2_truth > log2_min)
        assert valid.sum() < n0
        n0 = valid.sum()
        assert (log2_truth > log2_min).sum() > n0
        p_min = np.quantile(p_truth[valid], 0.5)
        valid = np.logical_and(valid, p_truth < p_min)
        assert valid.sum() < n0
        assert valid.sum() > 0
        n0 = valid.sum()
        assert (p_truth < p_min).sum() > n0

        q1_mask = np.logical_not(q1_truth < q1_min)
        qdiff_mask = np.logical_not(qdiff_truth < qdiff_min)
        log2_mask = np.logical_not(log2_truth < log2_min)
        p_mask = (p_truth < p_min)
        valid = np.logical_and(
            q1_mask,
            np.logical_and(
            qdiff_mask,
            np.logical_and(
                log2_mask,
                p_mask)))

        n0 = valid.sum()
        assert n0 > 0

        is_degenerate = False
        for n_combo in (2, 3):
            for mask_set in itertools.combinations(
                    [q1_mask, qdiff_mask, log2_mask, p_mask],
                    n_combo):
                this = mask_set[0]
                for ii in range(len(mask_set)):
                    this = np.logical_and(this, mask_set[ii])
                if not this.sum() > n0:
                    is_degenerate = True

        if is_degenerate:
            n_degenerate += 1

        n_tot += 1
        this = {'config': {
                    'q1_min_th': q1_min,
                    'qdiff_min_th': qdiff_min,
                    'log2_fold_min_th': log2_min,
                    'p_th': p_min},
                'expected': valid,
                'is_degenerate': is_degenerate}

        if not is_degenerate:
            non_degenerate_configs.append(this['config'])

        result[cl0][cl1] = [this]
        result[cl1][cl0] = [this]

    assert n_tot > n_degenerate

    # lookup mapping cluster pair to number of marker
    # genes; just to make sure we are building interesting
    # test cases
    idx_lookup = dict()

    # these will be configs that are applied to all pairs
    # (for use when testing find_markers_for_all_pairs)
    universal_configs = []

    for config in non_degenerate_configs:
        this_universal = {'config': config}
        this_expected = dict()
        for cl0, cl1 in itertools.combinations(cluster_name_list, 2):

            if cl0 not in idx_lookup:
                idx_lookup[cl0] = dict()
            if cl1 not in idx_lookup[cl0]:
                idx_lookup[cl0][cl1] = set()

            p_truth = p_value_fixture[cl0][cl1]
            penetrance_truth = penetrance_fixture[cl0][cl1]
            q1_truth = penetrance_truth['q1']
            qdiff_truth = penetrance_truth['qdiff']
            log2_truth = penetrance_truth['log2_fold']

            q1_min = config['q1_min_th']
            qdiff_min = config['qdiff_min_th']
            log2_min = config['log2_fold_min_th']
            p_min = config['p_th']

            q1_mask = np.logical_not(q1_truth < q1_min)
            qdiff_mask = np.logical_not(qdiff_truth < qdiff_min)
            log2_mask = np.logical_not(log2_truth < log2_min)
            p_mask = (p_truth < p_min)
            valid = np.logical_and(
                q1_mask,
                np.logical_and(
                qdiff_mask,
                np.logical_and(
                    log2_mask,
                    p_mask)))

            this = {'config': config,
                    'expected': valid,
                    'is_degenerate': True}

            result[cl0][cl1].append(this)
            result[cl1][cl0].append(this)
            valid_tuple = tuple(np.sort(valid))
            idx_lookup[cl0][cl1].add(valid_tuple)

            if cl0 not in this_expected:
                this_expected[cl0] = dict()
            if cl1 not in this_expected[cl0]:
                this_expected[cl0][cl1] = valid
        this_universal['expected'] = this_expected
        boring_t = boring_t_from_p_value(config['p_th'])

        if boring_t is not None:
            universal_configs.append(this_universal)


    # make sure each pair gets a diversity of 'expected' arrays
    for cl0 in idx_lookup:
        for cl1 in idx_lookup[cl0]:
            assert len(idx_lookup[cl0][cl1]) > 2

    assert len(universal_configs) > 0

    return result, universal_configs


@pytest.fixture(scope='module')
def threshold_mask_fixture(
        threshold_mask_generator_fixture):
    """
    A dict mapping cluster pairs to reference marker configs
    and the validity masks they produce (assuming that only minimume
    thresholds on penetrance stats are actually interesting)
    """
    return threshold_mask_generator_fixture[0]

@pytest.fixture(scope='module')
def threshold_mask_fixture_all_pairs(
        threshold_mask_generator_fixture):
    """
    A list of reference marker configs along with dicts mapping
    cluster pairs to the expected validity masks that go along with
    them (assuming only minimum penetrance thresholds are interesting)
    """
    return threshold_mask_generator_fixture[1]


@pytest.fixture(scope='module')
def precomputed_fixture(
        tmp_dir_fixture,
        taxonomy_tree_fixture,
        cluster_profile_fixture,
        n_genes):

    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='precomputed_stats_',
        suffix='.h5')

    n_clusters = len(cluster_profile_fixture)

    with h5py.File(h5_path, 'w') as dst:
        dst.create_dataset(
            'taxonomy_tree',
            data=taxonomy_tree_fixture.to_str().encode('utf-8'))

        dst.create_dataset(
            'col_names',
            data=json.dumps(
                [f'g_{ii}' for ii in range(n_genes)]).encode('utf-8'))

        dst.create_dataset(
            'sum', shape=(n_clusters, n_genes), dtype=float)
        dst.create_dataset(
            'sumsq', shape=(n_clusters, n_genes), dtype=float)
        dst.create_dataset(
            'ge1', shape=(n_clusters, n_genes), dtype=int)
        dst.create_dataset(
            'n_cells', shape=(n_clusters,), dtype=int)

        cluster_to_row = dict()
        for row_idx, cluster in enumerate(cluster_profile_fixture):
            cluster_to_row[cluster] = row_idx
            this = cluster_profile_fixture[cluster]
            dst['sum'][row_idx, :] = this['sum']
            dst['sumsq'][row_idx, :] = this['sumsq']
            dst['ge1'][row_idx, :] = this['ge1']
            dst['n_cells'][row_idx] = this['n_cells']
        dst.create_dataset(
            'cluster_to_row', data=json.dumps(cluster_to_row).encode('utf-8'))
    return h5_path
