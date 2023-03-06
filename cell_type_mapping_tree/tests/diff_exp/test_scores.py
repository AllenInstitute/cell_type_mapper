import pytest

import numpy as np

from hierarchical_mapping.diff_exp.scores import (
    aggregate_stats,
    score_differential_genes)


@pytest.fixture
def n_genes():
    return 26

@pytest.fixture
def leaf_node_fixture():
    """
    List of leaf nodes
    """
    return ['a', 'b', 'c', 'd',
            'e', 'f', 'g', 'h']


@pytest.fixture
def data_fixture(leaf_node_fixture, n_genes):
    """
    Fixture mapping leaf_node to a simulated
    gene expression matrix
    """
    rng = np.random.default_rng(76123412)
    result = dict()
    for node in leaf_node_fixture:
        n_cells = rng.integers(14, 37)
        data = 2.0*rng.random((n_cells, n_genes))
        zeroed_out = (rng.random((n_cells, n_genes))>0.85)
        data[zeroed_out] = 0.0
        result[node] = data
    return result

@pytest.fixture
def precomputed_stats_fixture(
        leaf_node_fixture,
        data_fixture,
        n_genes):
    """
    Fixture containing random data as
    dict mapping leaf node name to
        'n_cells'
        'sum'
        'sumsq'
        'gt0'
        'gt1'
    """
    result = dict()
    for node in leaf_node_fixture:
        these_stats = dict()
        data = data_fixture[node]
        n_cells = data.shape[0]
        these_stats['sum'] = data.sum(axis=0)
        these_stats['sumsq'] = (data**2).sum(axis=0)
        these_stats['gt0'] = (data>0).sum(axis=0)
        these_stats['gt1'] = (data>1).sum(axis=0)
        these_stats['n_cells'] = n_cells
        these_stats['data'] = data
        result[node] = these_stats

    return result


@pytest.mark.parametrize(
        "gt0_threshold, gt1_threshold",
        [(1, 0), (1, 40), (1, 1000)])
def test_aggregate_stats(
        data_fixture,
        precomputed_stats_fixture,
        leaf_node_fixture,
        n_genes,
        gt0_threshold,
        gt1_threshold):

    rng = np.random.default_rng(22312)
    population = rng.choice(leaf_node_fixture, 4, replace=False)

    actual = aggregate_stats(
                leaf_population=population,
                precomputed_stats=precomputed_stats_fixture,
                gt0_threshold=gt0_threshold,
                gt1_threshold=gt1_threshold)

    data_arr = [data_fixture[n] for n in population]
    data_arr = np.vstack(data_arr)

    assert actual['n_cells'] == data_arr.shape[0]

    expected_mean = np.mean(data_arr, axis=0)
    np.testing.assert_allclose(actual['mean'], expected_mean)

    expected_var = np.var(data_arr, axis=0, ddof=1)
    np.testing.assert_allclose(actual['var'], expected_var)

    expected_gt0 = (data_arr>0).sum(axis=0)
    expected_gt1 = (data_arr>1).sum(axis=0)

    expected_mask = np.logical_and(
        expected_gt0>=gt0_threshold,
        expected_gt1>=gt1_threshold)

    assert expected_mask.shape == (n_genes,)
    np.testing.assert_array_equal(actual['mask'], expected_mask)


@pytest.mark.parametrize(
        "gt0_threshold, gt1_threshold",
        [(1, 0), (1, 40), (1, 1000)])
def test_aggregate_stats(
        data_fixture,
        precomputed_stats_fixture,
        leaf_node_fixture,
        n_genes,
        gt0_threshold,
        gt1_threshold):
    """
    Just a smoketest to make sure we can run
    score_differential_genes
    """

    pop1 = ['a', 'c', 'd']
    pop2 = ['f', 'e', 'g', 'h']

    (score,
     validity) = score_differential_genes(
                     leaf_population_1=pop1,
                     leaf_population_2=pop2,
                     precomputed_stats=precomputed_stats_fixture,
                     gt1_threshold=gt1_threshold,
                     gt0_threshold=gt0_threshold)

    assert score.shape == (n_genes,)
    assert validity.shape == (n_genes,)
    assert validity.dtype == bool
    assert score.dtype == float
