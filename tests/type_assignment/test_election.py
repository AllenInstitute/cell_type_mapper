import pytest

import numpy as np
import os
from unittest.mock import patch

from cell_type_mapper.utils.torch_utils import (
    is_torch_available,
    use_torch)

from cell_type_mapper.type_assignment.election import (
    tally_votes,
    choose_node)


@pytest.mark.parametrize(
    "bootstrap_factor, bootstrap_iteration",
    [(0.7, 22),
     (0.4, 102),
     (0.9, 50),
     (1.0, 1)])
def test_tally_votes(
        bootstrap_factor,
        bootstrap_iteration):
    """
    Just a smoke test (does test output shape
    and that the total number of votes matches
    iterations)
    """
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

    reference_types = ['a', 'b', 'c', 'd']
    rng = np.random.default_rng(223112)
    mock_votes = np.array(
            [[2, 3, 1, 0],
             [4, 1, 0, 2],
             [0, 3, 5, 1],
             [4, 3, 1, 0]])
    mock_corr_sum = rng.random(mock_votes.shape, dtype=float)

    def dummy_tally_votes(*args, **kwargs):
        return (mock_votes, mock_corr_sum)

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
            bootstrap_iteration=5,
            n_choices=3,
            rng=None)

    expected_runners_up = [
        [('a', mock_corr_sum[0,0]/2),
         ('c', mock_corr_sum[0,2]/1)],
        [('d', mock_corr_sum[1, 3]/2),
         ('b', mock_corr_sum[1, 1]/1)],
        [('b', mock_corr_sum[2, 1]/3),
         ('d', mock_corr_sum[2, 3]/1)],
        [('b', mock_corr_sum[3, 1]/3),
         ('c', mock_corr_sum[3, 2]/1)]]

    assert len(runners_up) == len(expected_runners_up)
    for i_row in range(len(runners_up)):
        actual = runners_up[i_row]
        expected = expected_runners_up[i_row]
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert a[0] == e[0]
            np.testing.assert_allclose(a[1], e[1])
