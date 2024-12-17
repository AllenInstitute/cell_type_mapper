import pytest

import numpy as np

from cell_type_mapper.utils.utils import (
    _clean_up)

from cell_type_mapper.diff_exp.p_value_markers import (
    _get_validity_mask)


@pytest.fixture(scope='module')
def tmp_dir(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('p_mask_markers_')
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.mark.parametrize(
    "n_valid, valid_gene_idx, expected_markers",
    [(3,
      None,
      (6, 9, 11)),  # genes that absolutely pass penetrance test
     (5,
      None,
      (5, 6, 7, 9, 11)),
     # grab smallest 2 distances after absolute passing genes
     (5,
      np.array([2, 3, 5, 6, 11, 12, 13]),
      (5, 6, 11, 12, 13)),  # 2, 3 do not pass p-value test
     (5,
      np.array([2, 3, 5, 6, 7, 8, 10, 11]),
      (5, 6, 7, 8, 11)),  # do not need 10, 11 (distance too large)
     (5,
      np.array([2, 3, 5, 6, 8, 10, 11, 12, 13]),
      (5, 6, 8, 10, 11, 12)),  # degeneracy in distance between 10 and 12
     ]
)
def test_get_validity_mask(
        n_valid,
        valid_gene_idx,
        expected_markers):

    n_genes = 20
    gene_indices = np.arange(5, 16, dtype=int)
    raw_distances = np.arange(1, len(gene_indices)+1, dtype=float)

    raw_distances[7] = raw_distances[5]  # gene 12 == gene 10

    raw_distances[1] = -1.0  # gene 6
    raw_distances[4] = -1.0  # gene 9
    raw_distances[6] = -1.0  # gene 11

    actual = _get_validity_mask(
        n_valid=n_valid,
        n_genes=n_genes,
        gene_indices=gene_indices,
        raw_distances=raw_distances,
        valid_gene_idx=valid_gene_idx)

    expected = np.zeros(n_genes, dtype=bool)
    for ii in expected_markers:
        expected[ii] = True

    np.testing.assert_array_equal(expected, actual)
