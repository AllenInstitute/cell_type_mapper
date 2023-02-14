import pytest
from hierarchical_mapping.utils.utils import merge_index_list


@pytest.mark.parametrize(
    "input_list, expected",
    (([1,2,3,8,7], [(1,4), (7,9)]),
     ([3,7,2,1,8], [(1,4), (7,9)]),
     ([0,5,9,6,10,11,17], [(0,1), (5,7), (9,12), (17,18)]),
    ))
def test_merge_index_list(input_list, expected):
    actual = merge_index_list(input_list)
    assert actual == expected
