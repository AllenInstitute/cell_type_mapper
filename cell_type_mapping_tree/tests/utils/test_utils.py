import pytest
import numpy as np
from hierarchical_mapping.utils.utils import (
    merge_index_list,
    choose_int_dtype)


@pytest.mark.parametrize(
    "input_list, expected",
    (([1,2,3,8,7], [(1,4), (7,9)]),
     ([3,7,2,1,8], [(1,4), (7,9)]),
     ([0,5,9,6,10,11,17], [(0,1), (5,7), (9,12), (17,18)]),
    ))
def test_merge_index_list(input_list, expected):
    actual = merge_index_list(input_list)
    assert actual == expected



@pytest.mark.parametrize(
        "output_dtype",
        (np.uint8, np.int8, np.uint16, np.int16,
         np.uint32, np.int32, np.uint64, np.int64))
def test_choose_int_dtype(output_dtype):
    output_info = np.iinfo(output_dtype)
    min_val = (output_info.min)+0.1
    max_val = float(output_info.max)-0.1
    assert choose_int_dtype((min_val, max_val)) == output_dtype
