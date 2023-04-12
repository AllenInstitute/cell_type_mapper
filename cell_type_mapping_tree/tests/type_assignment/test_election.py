import pytest

import numpy as np
from scipy.spatial.distance import cdist as scipy_cdist
from unittest.mock import patch

from hierarchical_mapping.type_assignment.election import (
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

    result = tally_votes(
        query_gene_data=query_data,
        reference_gene_data=reference_data,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    assert result.shape == (n_query, n_baseline)
    for i_row in range(n_query):
        assert result[i_row, :].sum() == bootstrap_iteration


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
     confidence) = choose_node(
        query_gene_data=query_data,
        reference_gene_data=reference_data,
        reference_types=reference_types,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    assert len(result) == n_query
    assert len(confidence) == n_query


def test_confidence_result():
    """
    Test that types are correctly chosen
    and confidence correctly reported
    """

    reference_types = ['a', 'b', 'c']

    def dummy_tally_votes(*args, **kwargs):
        return np.array(
            [[1, 3, 1],
             [4, 1, 0],
             [0, 0, 5],
             [4, 0, 1]])

    to_replace = 'hierarchical_mapping.type_assignment.election.tally_votes'
    with patch(to_replace, new=dummy_tally_votes):
        (results,
         confidence) = choose_node(
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
        [0.6, 0.8, 1.0, 0.8])


def test_cosine_v_corr_votes():
    """
    test, using boostrap_factor=1.0, bootstrap_iteration=1, that
    tally_votes correctly propagates distance metrics
    """

    # assemble ground truth with scipy_cdist
    rng = np.random.default_rng(88123)
    query = rng.random((56, 17))
    reference = rng.random((123, 17))

    corr_dist = scipy_cdist(
        reference,
        query,
        metric='correlation')

    cos_dist = scipy_cdist(
        reference,
        query,
        metric='cosine')

    corr_nn = np.argmin(corr_dist, axis=0)
    cos_nn = np.argmin(cos_dist, axis=0)
    assert corr_nn.shape == cos_nn.shape
    assert corr_nn.shape == (query.shape[0],)
    assert not np.allclose(corr_nn, cos_nn)

    cos_expected = np.zeros((query.shape[0], reference.shape[0]),
                            dtype=int)
    corr_expected = np.zeros((query.shape[0], reference.shape[0]),
                             dtype=int)

    for ii in range(query.shape[0]):
        cos_expected[ii, cos_nn[ii]] = 1
        corr_expected[ii, corr_nn[ii]] = 1

    # get actual votes

    cos_votes = tally_votes(
        query_gene_data=query,
        reference_gene_data=reference,
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        rng=rng,
        distance_metric='cos')

    np.testing.assert_array_equal(cos_votes, cos_expected)

    corr_votes = tally_votes(
        query_gene_data=query,
        reference_gene_data=reference,
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        rng=rng,
        distance_metric='corr')

    np.testing.assert_array_equal(corr_votes, corr_expected)


    # nw test choose_node
    (cos_node,
     _) = choose_node(
        query_gene_data=query,
        reference_gene_data=reference,
        reference_types=[str(ii) for ii in range(reference.shape[0])],
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        rng=rng,
        distance_metric='cos')

    expected = np.array([str(ii) for ii in cos_nn])
    np.testing.assert_array_equal(expected, cos_node)


    (corr_node,
     _) = choose_node(
        query_gene_data=query,
        reference_gene_data=reference,
        reference_types=[str(ii) for ii in range(reference.shape[0])],
        bootstrap_factor=1.0,
        bootstrap_iteration=1,
        rng=rng,
        distance_metric='corr')

    expected = np.array([str(ii) for ii in corr_nn])
    np.testing.assert_array_equal(expected, corr_node)
