import pytest

import numpy as np

from cell_type_mapper.corr.utils import (
    match_genes)


@pytest.mark.parametrize(
        "query, expected",
        [(['b', 'x', 'f', 'e', 'w'],
          {'reference': np.array([0, 5, 3]),
           'query': np.array([0, 3, 2])}),
         (['w', 'x', 'y'],
            {'reference': [],
             'query': []}
          ),
         (['x', 'f', 'e', 'w', 'b'],
          {'reference': np.array([0, 5, 3]),
           'query': np.array([4, 2, 1])}),
         ]
)
def test_match_genes(
        query,
        expected):
    reference_gene_names = ['b', 'a', 'c', 'f', 'd', 'e']
    actual = match_genes(
                reference_gene_names=reference_gene_names,
                query_gene_names=query)

    if len(expected['reference']) > 0:
        np.testing.assert_array_equal(actual['reference'],
                                      expected['reference'])
        np.testing.assert_array_equal(actual['query'],
                                      expected['query'])

        q_names = [query[idx] for idx in actual['query']]
        r_names = [reference_gene_names[idx] for idx in actual['reference']]
        assert q_names == r_names
        assert q_names == actual['names']

    else:
        assert len(actual['reference']) == 0
        assert len(actual['query']) == 0
        assert len(actual['names']) == 0


def test_match_genes_with_markers():
    result = match_genes(
        reference_gene_names=['a', 'e', 'b', 'c', 'd', 'f'],
        query_gene_names=['b', 'f', 'e', 'h', 'i', 'c'],
        marker_gene_names=['e', 'c'])

    np.testing.assert_array_equal(
        result['reference'],
        np.array([3, 1]))
    np.testing.assert_array_equal(
        result['query'],
        np.array([5, 2]))

    assert result['names'] == ['c', 'e']

    result = match_genes(
        reference_gene_names=['a', 'e', 'b', 'c', 'd', 'f'],
        query_gene_names=['b', 'f', 'e', 'h', 'i', 'c'],
        marker_gene_names=['e', 'c', 'x'])

    np.testing.assert_array_equal(
        result['reference'],
        np.array([3, 1]))
    np.testing.assert_array_equal(
        result['query'],
        np.array([5, 2]))

    assert result['names'] == ['c', 'e']

    result = match_genes(
        reference_gene_names=['a', 'e', 'b', 'c', 'd', 'f'],
        query_gene_names=['b', 'f', 'e', 'h', 'i', 'c'],
        marker_gene_names=['x', 'y'])

    np.testing.assert_array_equal(
        result['reference'],
        np.array([]))
    np.testing.assert_array_equal(
        result['query'],
        np.array([]))
    assert result['names'] == []
