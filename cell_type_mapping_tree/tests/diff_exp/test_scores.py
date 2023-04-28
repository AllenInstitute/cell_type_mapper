import pytest

import numpy as np

from hierarchical_mapping.diff_exp.scores import (
    aggregate_stats,
    score_differential_genes,
    diffexp_score,
    rank_genes)


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



def test_aggregate_stats(
        data_fixture,
        precomputed_stats_fixture,
        leaf_node_fixture,
        n_genes):

    rng = np.random.default_rng(22312)
    population = rng.choice(leaf_node_fixture, 4, replace=False)

    actual = aggregate_stats(
                leaf_population=population,
                precomputed_stats=precomputed_stats_fixture)

    data_arr = [data_fixture[n] for n in population]
    data_arr = np.vstack(data_arr)

    assert actual['n_cells'] == data_arr.shape[0]

    expected_mean = np.mean(data_arr, axis=0)
    np.testing.assert_allclose(actual['mean'], expected_mean)

    expected_var = np.var(data_arr, axis=0, ddof=1)
    np.testing.assert_allclose(actual['var'], expected_var)

    expected_gt0 = (data_arr>0).sum(axis=0)
    expected_gt1 = (data_arr>1).sum(axis=0)

    np.testing.assert_array_equal(actual['gt0'], expected_gt0)
    np.testing.assert_array_equal(actual['gt1'], expected_gt1)


def test_score_diff_smoke(
        data_fixture,
        precomputed_stats_fixture,
        leaf_node_fixture,
        n_genes):
    """
    Just a smoketest to make sure we can run
    score_differential_genes
    """

    pop1 = ['a', 'c', 'd']
    pop2 = ['f', 'e', 'g', 'h']

    (score,
     validity,
     up_mask) = score_differential_genes(
                     leaf_population_1=pop1,
                     leaf_population_2=pop2,
                     precomputed_stats=precomputed_stats_fixture)

    assert score.shape == (n_genes,)
    assert validity.shape == (n_genes,)
    assert validity.dtype == bool
    assert score.dtype == float
    assert up_mask.shape == (n_genes,)
    assert up_mask.dtype == np.uint8


def test_diffexp_score():
    """
    Just make sure that the score is symmetric
    (i.e. the gene will get a high score, regardless
    of which population it is expressed in)
    """

    # gene 1 is the best discriminator
    scores = diffexp_score(
        mean1=np.array([1.0, 0.0]),
        var1=np.array([0.1, 0.05]),
        n1=24,
        mean2=np.array([1.0, 0.5]),
        var2=np.array([0.2, 0.1]),
        n2=35)

    assert scores[1] > scores[0]

    # gene 0 is the bet discriminator
    scores = diffexp_score(
        mean1=np.array([1.0, 0.0]),
        var1=np.array([0.1, 0.05]),
        n1=24,
        mean2=np.array([0.0, 0.0]),
        var2=np.array([0.2, 0.1]),
        n2=35)

    assert scores[0] > scores[1]


@pytest.mark.parametrize(
    "scores, validity, expected",
    [(np.array([0.0, 2.0, 1.0, 2.0, 3.0]),
      np.array([True, False, True, True, False]),
      np.array([3, 2, 0, 4, 1])),
     (np.array([11.0, 0.0, 22.0, 17.0, 8.0]),
      np.array([False, True, True, True, False]),
      np.array([2, 3, 1, 0, 4]))
    ])
def test_rank_genes(
        scores,
        validity,
        expected):

    actual = rank_genes(
                scores=scores,
                validity=validity)
    np.testing.assert_array_equal(expected, actual)
