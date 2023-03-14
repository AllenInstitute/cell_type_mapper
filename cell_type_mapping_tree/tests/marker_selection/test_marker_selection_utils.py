import pytest
import numpy as np

from hierarchical_mapping.marker_selection.utils import (
    _process_rank_chunk)



@pytest.mark.parametrize(
    "valid_rows, valid_genes, genes_per_pair, expected",
    [(set([2, 6, 7, 11]),
      set(np.arange(5, 27)),
      3,
      set([8, 22, 6, 9, 13])),
     (set([2, 6, 7, 11]),
      set(np.arange(5, 27)),
      2,
      set([8, 22, 6, 9])),
     (set([2, 6, 7, 11]),
      set([99, 98, 100]),
      3,
      set([])),
     (set([400, 401, 402]),
      set(np.arange(5, 27)),
      3,
      set([]))
    ])
def test_process_rank_chunk(
        valid_rows,
        valid_genes,
        genes_per_pair,
        expected):

    row0 = 4
    row1 = 15
    rank_chunk = np.ones((row1-row0, 14), dtype=int)
    rank_chunk[2][2] = 6
    rank_chunk[2][7] = 9
    rank_chunk[2][8] = 13
    rank_chunk[2][10] = 17

    # this row should not be valid
    rank_chunk[4][1] = 18
    rank_chunk[4][2] = 19
    rank_chunk[4][3] = 20

    rank_chunk[7][11] = 8
    rank_chunk[7][12] = 22

    actual = _process_rank_chunk(
                valid_rows=valid_rows,
                valid_genes=valid_genes,
                rank_chunk=rank_chunk,
                row0=row0,
                row1=row1,
                genes_per_pair=genes_per_pair)

    assert actual == expected
