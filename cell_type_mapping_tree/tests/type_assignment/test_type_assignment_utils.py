import pytest

import numpy as np

from hierarchical_mapping.type_assignment.utils import (
    choose_node)


@pytest.mark.parametrize(
    "bootstrap_factor, bootstrap_iteration",
    [(0.7, 22),
     (0.4, 102),
     (0.9, 50)])
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

    result = choose_node(
        query_gene_data=query_data,
        reference_gene_data=reference_data,
        reference_types=reference_types,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    assert len(result) == n_query
