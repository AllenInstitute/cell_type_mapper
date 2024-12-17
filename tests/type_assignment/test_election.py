import pytest

import numpy as np
import os
import pathlib
from unittest.mock import patch
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    use_torch)

from cell_type_mapper.type_assignment.election import (
    tally_votes,
    choose_node,
    run_type_assignment,
    aggregate_votes)

from cell_type_mapper.type_assignment.marker_cache_v2 import (
    create_marker_cache_from_specified_markers)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('election_test_dir_'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.mark.parametrize(
    "bootstrap_factor, bootstrap_iteration, with_torch",
    [(0.7, 22, True),
     (0.4, 102, True),
     (0.9, 50, True),
     (1.0, 1, True),
     (0.7, 22, False),
     (0.4, 102, False),
     (0.9, 50, False),
     (1.0, 1, False)])
def test_tally_votes(
        bootstrap_factor,
        bootstrap_iteration,
        with_torch):
    """
    Just a smoke test (does test output shape
    and that the total number of votes matches
    iterations)
    """
    if with_torch:
        if not is_torch_available():
            return
        env_var = 'AIBS_BKP_USE_TORCH'
        os.environ[env_var] = 'true'
        assert use_torch()

    rng = np.random.default_rng(776123)

    n_genes = 25
    n_query = 64
    n_baseline = 222

    query_data = rng.random((n_query, n_genes))
    reference_data = rng.random((n_baseline, n_genes))

    (votes,
     corr_sum) = tally_votes(
        query_gene_data=query_data,
        reference_gene_data=reference_data,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    assert votes.shape == (n_query, n_baseline)
    assert corr_sum.shape == (n_query, n_baseline)
    for i_row in range(n_query):
        assert votes[i_row, :].sum() == bootstrap_iteration
    assert corr_sum.max() > 1.0e-6

    if with_torch:
        os.environ[env_var] = ''
        assert not use_torch()


def test_tally_votes_mocked_result():
    """
    Use a mock to control which neighbors are
    chosen with which correlation values at each
    iteration. Test that tally votes correctly sums
    them.
    """

    n_query = 5
    n_reference = 8
    n_genes = 12

    mock_votes = [
        [1, 3, 4, 4, 6],
        [1, 3, 4, 1, 0],
        [5, 3, 6, 4, 1],
        [1, 2, 6, 4, 1]
    ]

    mock_corr = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.3, 0.6, 0.9, 1.2, 1.5],
        [0.4, 0.8, 1.2, 1.6, 2.0]
    ]

    def dummy_nn(*args, **kwargs):
        if not hasattr(dummy_nn, 'ct'):
            dummy_nn.ct = -1
        dummy_nn.ct += 1
        return (np.array(mock_votes[dummy_nn.ct]),
                np.array(mock_corr[dummy_nn.ct]))

    to_replace = 'cell_type_mapper.utils.distance_utils.'
    to_replace += 'correlation_nearest_neighbors'
    with patch(to_replace, new=dummy_nn):
        (actual_votes,
         actual_corr_sum) = tally_votes(
                 query_gene_data=np.zeros((n_query, n_genes), dtype=float),
                 reference_gene_data=np.zeros((n_reference, n_genes), dtype=float),
                 bootstrap_factor=0.5,
                 bootstrap_iteration=4,
                 rng=np.random.default_rng(2213))

    expected_votes = np.zeros((n_query, n_reference), dtype=int)
    expected_corr = np.zeros((n_query, n_reference), dtype=float)
    for votes, corr in zip(mock_votes, mock_corr):
        for ii in range(n_query):
            jj = votes[ii]
            expected_votes[ii, jj] += 1
            expected_corr[ii, jj] += corr[ii]

    np.testing.assert_array_equal(actual_votes, expected_votes)
    np.testing.assert_allclose(
        actual_corr_sum,
        expected_corr,
        atol=0.0,
        rtol=1.0e-6)


@pytest.mark.skipif(not is_torch_available(), reason='no torch')
@pytest.mark.parametrize(
    "bootstrap_factor, bootstrap_iteration",
    [(0.7, 22),
     (0.4, 102),
     (0.9, 50),
     (1.0, 1)])
def test_tally_votes_gpu(
        bootstrap_factor,
        bootstrap_iteration):
    """
    Test that CPU and GPU implementations of tally_votes give the
    same result
    """
    env_var = 'AIBS_BKP_USE_TORCH'
    rng = np.random.default_rng(55666)

    n_genes = 25
    n_query = 64
    n_baseline = 222

    query_data = rng.random((n_query, n_genes))
    reference_data = rng.random((n_baseline, n_genes))

    os.environ[env_var] = 'false'
    assert not use_torch()
    rng_seed = 1122334455
    rng = np.random.default_rng(rng_seed)
    cpu_result = tally_votes(
        query_gene_data=query_data,
        reference_gene_data=reference_data,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    os.environ[env_var] = 'true'
    assert use_torch()
    rng = np.random.default_rng(rng_seed)
    gpu_result = tally_votes(
        query_gene_data=query_data,
        reference_gene_data=reference_data,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    os.environ[env_var] = ''

    np.testing.assert_array_equal(
        cpu_result[0],
        gpu_result[0])

    np.testing.assert_allclose(
        cpu_result[1],
        gpu_result[1],
        atol=0.0,
        rtol=1.0e-5)

    assert cpu_result[1].max() > 1.0e-6

@pytest.mark.parametrize(
    "bootstrap_factor, bootstrap_iteration",
    [(0.7, 22),
     (0.4, 102),
     (0.9, 50),
     (1.0, 1)])
def test_choose_node_smoke(
        bootstrap_factor,
        bootstrap_iteration):
    """
    Just a smoke test
    """
    rng = np.random.default_rng(776123)

    n_genes = 25
    n_query = 64
    n_baseline = 222

    query_data = rng.random((n_query, n_genes))
    reference_data = rng.random((n_baseline, n_genes))
    reference_types = [f"type_{ii}" for ii in range(n_baseline)]

    (result,
     confidence,
     avg_corr,
     _) = choose_node(
        query_gene_data=query_data,
        reference_gene_data=reference_data,
        reference_types=reference_types,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    assert result.shape == (n_query,)
    assert confidence.shape == (n_query,)
    assert avg_corr.shape == (n_query,)


def test_confidence_result():
    """
    Test that types are correctly chosen
    and confidence correctly reported
    """

    reference_types = ['a', 'b', 'c']
    rng = np.random.default_rng(223112)
    mock_votes = np.array(
            [[1, 3, 1],
             [4, 1, 0],
             [0, 0, 5],
             [4, 0, 1]])
    mock_corr_sum = rng.random(mock_votes.shape, dtype=float)

    def dummy_tally_votes(*args, **kwargs):
        return (mock_votes, mock_corr_sum)

    to_replace = 'cell_type_mapper.type_assignment.election.tally_votes'
    with patch(to_replace, new=dummy_tally_votes):
        (results,
         confidence,
         avg_corr,
         _) = choose_node(
            query_gene_data=None,
            reference_gene_data=None,
            reference_types=reference_types,
            bootstrap_factor=None,
            bootstrap_iteration=5,
            rng=None)

    np.testing.assert_array_equal(
        results, ['b', 'a', 'c', 'a'])

    np.testing.assert_allclose(
        confidence,
        [0.6, 0.8, 1.0, 0.8],
        atol=0.0,
        rtol=1.0e-6)

    expected_avg_corr = np.array(
        [mock_corr_sum[0, 1]/3,
         mock_corr_sum[1, 0]/4,
         mock_corr_sum[2, 2]/5,
         mock_corr_sum[3, 0]/4])

    np.testing.assert_allclose(
        avg_corr,
        expected_avg_corr,
        atol=0.0,
        rtol=1.0e-6)


def test_runners_up():
    """
    Test that choose_node correctly selects the
    N runners up cell types.
    """

    reference_types = ['a', 'b', 'c', 'd', 'e']
    rng = np.random.default_rng(223112)
    mock_votes = np.array(
            [[7, 10, 6, 0, 0],
             [11, 4, 0, 5, 0],
             [0, 9, 22, 7, 2],
             [44, 11, 6, 0, 0]])
    mock_corr_sum = rng.random(mock_votes.shape, dtype=float)

    def dummy_tally_votes(*args, **kwargs):
        return (mock_votes, mock_corr_sum)

    bootstrap_iteration = 16

    to_replace = 'cell_type_mapper.type_assignment.election.tally_votes'
    with patch(to_replace, new=dummy_tally_votes):
        (results,
         confidence,
         avg_corr,
         runners_up) = choose_node(
            query_gene_data=None,
            reference_gene_data=None,
            reference_types=reference_types,
            bootstrap_factor=None,
            bootstrap_iteration=bootstrap_iteration,
            n_assignments=4,
            rng=None)

    # note: choose_node gets the denominator for bootstrapping
    # probability from bootstrap_iteration, hence the uniform
    # denominators in the 2nd element of the tuples below
    expected_runners_up = [
        [('a', mock_corr_sum[0,0]/7, 7.0/bootstrap_iteration),
         ('c', mock_corr_sum[0, 2]/6, 6.0/bootstrap_iteration)],
        [('d', mock_corr_sum[1, 3]/5, 5.0/bootstrap_iteration),
         ('b', mock_corr_sum[1, 1]/4, 4.0/bootstrap_iteration)],
        [('b', mock_corr_sum[2, 1]/9, 9.0/bootstrap_iteration),
         ('d', mock_corr_sum[2, 3]/7, 7.0/bootstrap_iteration),
         ('e', mock_corr_sum[2, 4]/2, 2.0/bootstrap_iteration)],
        [('b', mock_corr_sum[3, 1]/11, 11.0/bootstrap_iteration),
         ('c', mock_corr_sum[3, 2]/6, 6.0/bootstrap_iteration)]]

    assert len(runners_up) == len(expected_runners_up)
    ct_false = 0
    for i_row in range(len(runners_up)):
        actual = runners_up[i_row]
        expected = expected_runners_up[i_row]
        assert len(actual) == 3
        for idx in range(len(expected)):
            a = actual[idx]
            e = expected[idx]
            assert a[0] == e[0]
            np.testing.assert_allclose(a[2], e[1])
            np.testing.assert_allclose(a[3], e[2])
            assert a[1]

        # any runners up that received no votes should be
        # marked with 'False' validity flag.
        if len(expected) != len(actual):
            for idx in range(len(expected), len(actual)):
                assert not actual[idx][1]
                ct_false += 1
    assert ct_false > 0


def test_run_type_assignment(
        tmp_dir_fixture):
    """
    Test the outputs of run_type_assignment. Chiefly,
    test that the runners up are, in fact, descendants
    of the parents they are supposed to descend from,
    and that NULL runners up are handled correctly.

    At the end, it also tests that avg_correlation is properly
    backfilled in cases where there is only one descendant
    in the taxonomy tree.
    """

    rng = np.random.default_rng(865211)

    n_clusters = 15
    tree_data = {
        'hierarchy': ['l1', 'l2', 'cluster'],
        'l1': {
            'l1a': ['l2a', 'l2c'],
            'l1b': ['l2b'],
            'l1c': ['l2e', 'l2d']
        },
        'l2': {
            'l2a': ['c0'],
            'l2b': ['c1'],
            'l2c': ['c5', 'c7', 'c10', 'c11', 'c3', 'c4', 'c12'],
            'l2d': ['c6'],
            'l2e': ['c8', 'c9', 'c2', 'c13', 'c14']
        },
        'cluster': {
            f'c{ii}': [] for ii in range(n_clusters)
        }
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        taxonomy_tree = TaxonomyTree(data=tree_data)

    reference_gene_names = [f'gene_{ii}' for ii in range(40)]

    marker_lookup = {
        'None': ['gene_0', 'gene_1', 'gene_2', 'gene_24', 'gene_25'],
        'l1/l1a': ['gene_3', 'gene_4', 'gene_19', 'gene_20'],
        'l1/l1c': ['gene_5', 'gene_6', 'gene_7', 'gene_21'],
        'l2/l2b': ['gene_8', 'gene_9', 'gene_10', 'gene_22'],
        'l2/l2c': ['gene_11', 'gene_12', 'gene_17', 'gene_18'],
        'l2/l2e': ['gene_13', 'gene_14', 'gene_15', 'gene_23']
    }

    cluster_to_gene = np.zeros((n_clusters, len(reference_gene_names)), dtype=float)
    cluster_name_list = list(tree_data['cluster'].keys())
    cluster_name_list.sort()
    for i_cluster, cluster_name in enumerate(cluster_name_list):
        parents = taxonomy_tree.parents(level='cluster', node=cluster_name)
        these_genes = [0, 1, 2]
        for level in parents:
            parent_key = f'{level}/{parents[level]}'
            if parent_key in marker_lookup:
                these_genes += [int(g.replace('gene_','')) for g in marker_lookup[parent_key]]
        these_genes = list(set(these_genes))
        these_genes.sort()
        signal = np.zeros(len(reference_gene_names))
        signal[these_genes] = 2.0*rng.random(len(these_genes)) + 1.0
        cluster_to_gene[i_cluster,: ] = signal

    reference_data = CellByGeneMatrix(
        data=cluster_to_gene,
        gene_identifiers=reference_gene_names,
        cell_identifiers=cluster_name_list,
        normalization="log2CPM")

    n_cells = 200
    query_data = np.zeros((n_cells, len(reference_gene_names)), dtype=float)
    for i_cell in range(n_cells):
        i_cluster = i_cell % n_clusters
        signal = cluster_to_gene[i_cluster, :]
        signal = signal * (rng.random(len(reference_gene_names))*0.3+0.8)
        noise = (0.5-rng.random(len(reference_gene_names)))*2.0
        query_data[i_cell, :] = signal + noise

    query_data = CellByGeneMatrix(
        data=query_data,
        gene_identifiers=reference_gene_names,
        normalization="log2CPM")

    marker_cache_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='marker_cache_',
        suffix='.h5')

    create_marker_cache_from_specified_markers(
        marker_lookup=marker_lookup,
        reference_gene_names=reference_gene_names,
        query_gene_names=reference_gene_names,
        output_cache_path=marker_cache_path)

    factor = 0.6666
    bootstrap_factor_lookup = {
        level: factor
        for level in taxonomy_tree.hierarchy
    }
    bootstrap_factor_lookup['None'] = factor

    results = run_type_assignment(
        full_query_gene_data=query_data,
        leaf_node_matrix=reference_data,
        marker_gene_cache_path=marker_cache_path,
        taxonomy_tree=taxonomy_tree,
        bootstrap_factor_lookup=bootstrap_factor_lookup,
        bootstrap_iteration=30,
        rng=rng)

    more_than_one_runner_up = 0
    for cell in results:
        for level in cell:
            if level == 'cell_id':
                continue
            this_level = cell[level]
            family_tree = taxonomy_tree.parents(
                level=level,
                node=this_level['assignment'])
            n_runners_up = len(this_level['runner_up_assignment'])
            assert len(this_level['runner_up_correlation']) == n_runners_up
            assert len(this_level['runner_up_probability']) == n_runners_up
            if n_runners_up == 0:
                # check that assignment was unanimous (either because it
                # was or because there was only one child to choose at this
                # level)
                np.testing.assert_allclose(
                    this_level['bootstrapping_probability'],
                    1.0,
                    atol=0.0,
                    rtol=1.0e-6)
            else:
                if n_runners_up > 1:
                    more_than_one_runner_up += 1

                    # check that runners up are ordered by probability
                    for ir in range(1, n_runners_up, 1):
                        r0 = this_level['runner_up_probability'][ir]
                        r1 = this_level['runner_up_probability'][ir-1]
                        assert r0 > 0.0
                        assert r0 <= r1

                assert this_level['runner_up_probability'][0] <= this_level['bootstrapping_probability']

                # check that probability sums to <= 1.0
                assert this_level['bootstrapping_probability'] < 1.0
                p_sum = this_level['bootstrapping_probability'] + sum(this_level['runner_up_probability'])
                eps = 1.0e-6
                assert p_sum <= (1.0+eps)

                # check that runners up have the same parentage
                # as the assigned node
                for ru in this_level['runner_up_assignment']:
                    if level == taxonomy_tree.leaf_level:
                        # Note: at higher than the leaf level it is possible for
                        # the same level to appear as a runner up
                        assert ru != this_level['assignment']

                    other_tree = taxonomy_tree.parents(
                        level=level,
                        node=ru)

                    assert other_tree == family_tree

    assert more_than_one_runner_up > 0

    # look for cases where there was only one valid descendant
    # in the taxonomy tree
    n_l1b = 0
    n_l2d = 0
    for cell in results:
        if cell['l1']['assignment'] == 'l1b':
            np.testing.assert_allclose(
                cell['l2']['avg_correlation'],
                cell['l1']['avg_correlation'],
                atol=0.0,
                rtol=1.0e-6)
            np.testing.assert_allclose(
                cell['l2']['bootstrapping_probability'],
                1.0,
                atol=0.0,
                rtol=1.0e-6)
            assert cell['l2']['assignment'] == 'l2b'
            np.testing.assert_allclose(
                cell['cluster']['avg_correlation'],
                cell['l1']['avg_correlation'],
                atol=0.0,
                rtol=1.0e-6)
            np.testing.assert_allclose(
                cell['cluster']['bootstrapping_probability'],
                1.0,
                atol=0.0,
                rtol=1.0e-6)
            assert cell['cluster']['assignment'] == 'c1'
            n_l1b += 1
        if cell['l2']['assignment'] == 'l2d':
            np.testing.assert_allclose(
                cell['cluster']['avg_correlation'],
                cell['l2']['avg_correlation'],
                atol=0.0,
                rtol=1.0e-6)
            np.testing.assert_allclose(
                cell['cluster']['bootstrapping_probability'],
                1.0,
                atol=0.0,
                rtol=1.0e-6)
            assert cell['cluster']['assignment'] == 'c6'
            n_l2d += 1

    assert n_l1b > 2
    assert n_l2d > 2

    # verify that aggregate probability is correct
    expected = []
    actual = []
    for cell in results:
        prob = 1.0
        for level in taxonomy_tree.hierarchy:
            prob *= cell[level]['bootstrapping_probability']
            expected.append(prob)
            actual.append(cell[level]['aggregate_probability'])

    np.testing.assert_allclose(
        expected,
        actual,
        atol=0.0,
        rtol=1.0e-6)


def test_aggregate_votes():

    rng = np.random.default_rng(2213)

    reference_types = ['b', 'b', 'a', 'd', 'a', 'c', 'b']
    
    n_query = 17
    votes = rng.integers(2, 15, (n_query, len(reference_types)))
    corr = rng.random((n_query, len(reference_types)))

    (new_votes,
     new_corr,
     new_ref) = aggregate_votes(
         vote_array=votes,
         correlation_array=corr,
         reference_types=reference_types)

    assert set(new_ref) == set(['a', 'b', 'c', 'd'])
    assert len(new_ref) == 4

    type_to_idx = {t: ii for ii, t in enumerate(new_ref)}

    expected_votes = votes[:, 0] + votes[:, 1] + votes[:, 6]
    actual_votes = new_votes[:, type_to_idx['b']]
    np.testing.assert_array_equal(expected_votes, actual_votes)

    expected_corr = corr[:, 0] + corr[:, 1] + corr[:, 6]
    actual_corr = new_corr[:, type_to_idx['b']]
    np.testing.assert_allclose(
        expected_corr, actual_corr,
        atol=0.0, rtol=1.0e-6)

    expected_votes = votes[:, 2] + votes[:, 4]
    actual_votes = new_votes[:, type_to_idx['a']]
    np.testing.assert_array_equal(expected_votes, actual_votes)

    expected_corr = corr[:, 4] + corr[:, 2]
    actual_corr = new_corr[:, type_to_idx['a']]
    np.testing.assert_allclose(
        expected_corr, actual_corr,
        atol=0.0, rtol=1.0e-6)

    np.testing.assert_array_equal(
        new_votes[:, type_to_idx['d']],
        votes[:, 3])
    np.testing.assert_allclose(
        new_corr[:, type_to_idx['d']],
        corr[:, 3],
        atol=0.0,
        rtol=1.0e-6)

    np.testing.assert_array_equal(
        new_votes[:, type_to_idx['c']],
        votes[:, 5])
    np.testing.assert_allclose(
        new_corr[:, type_to_idx['c']],
        corr[:, 5],
        atol=0.0,
        rtol=1.0e-6)
