import pytest

import numpy as np

from hierarchical_mapping.corr.utils import (
    match_genes)


@pytest.mark.parametrize(
        "query, expected",
        [(['b', 'x', 'f', 'e', 'w'],
          {'reference': np.ndarray([1, 4, 5]),
           'query': np.ndarray([0, 3, 2])}),
          (['w', 'x', 'y'],
           {'reference': [],
            'query': []}
          )
        ])
def test_match_genes(
        query,
        expected):
    reference_gene_names = ['a', 'b', 'c', 'd', 'e', 'f']
    actual = match_genes(
                reference_gene_names=reference_gene_names,
                query_gene_names=query)

    if len(expected['reference']) > 0:
        np.testing.assert_array_equal(actual['reference'],
                                      expected['reference'])
        np.testing.assert_array_equal(actual['query'],
                                      expected['query'])
    else:
        assert len(actual['reference']) == 0
        assert len(actual['query']) == 0
