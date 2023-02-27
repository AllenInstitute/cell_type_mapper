import pytest
import numpy as np
import copy
from hierarchical_mapping.utils.utils import (
    merge_index_list,
    refactor_row_chunk_list)


@pytest.mark.parametrize(
    "input_list, expected",
    (([1,2,3,8,7], [(1,4), (7,9)]),
     ([3,7,2,1,8], [(1,4), (7,9)]),
     ([0,5,9,6,10,11,17], [(0,1), (5,7), (9,12), (17,18)]),
    ))
def test_merge_index_list(input_list, expected):
    actual = merge_index_list(input_list)
    assert actual == expected



@pytest.mark.parametrize("final_chunk_size", [3, 5, 7, 11])
def test_refactor_row_chunk_list(final_chunk_size):

    input_chunks = [
        [1, 14, 9, 4],
        [6, 2, 8],
        [13, 5],
        [4]]

    output_chunks = refactor_row_chunk_list(
                        row_chunk_list=copy.deepcopy(input_chunks),
                        final_chunk_size=final_chunk_size)

    # make sure input was not changed
    assert input_chunks == [[1, 14, 9, 4],
                            [6, 2, 8],
                            [13, 5],
                            [4]]

    for r in output_chunks:
        assert len(r) > 0

    for ii in range(len(output_chunks)-1):
        assert len(output_chunks[ii]) % final_chunk_size == 0

    assert len(output_chunks[-1]) <= final_chunk_size

    input_flat = []
    for r in input_chunks:
        input_flat += r
    output_flat = []
    for r in output_chunks:
        output_flat += r

    print(output_chunks)
    assert output_flat == input_flat
