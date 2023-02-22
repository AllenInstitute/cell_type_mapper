import pytest

from hierarchical_mapping.utils.zarr_wrappers import (
    ZarrCacheWrapper,
    separate_into_chunks)


def test_zarr_cache_wrapper():
    pass


def test_separate_into_chunks_qc():
    """
    Test that error is raised when index_list is not contiguous
    """
    index_list = [1,2,3,7]
    with pytest.raises(RuntimeError, match="not contiguous"):
        separate_into_chunks(index_list, 100)


def test_separate_into_chunks():

    index_list = [5, 6, 7, 8, 9]
    actual = separate_into_chunks(
                index_list=index_list,
                chunk_shape=[3,])

    expected = [{'chunk_spec': ((3, 6),),
                 'output_loc': (0, 1),
                 'chunk_loc': (2, 3)},
                {'chunk_spec': ((6, 9),),
                 'output_loc': (1, 4),
                 'chunk_loc': (0, 3)},
                {'chunk_spec': ((9, 12),),
                 'output_loc': (4, 5),
                 'chunk_loc': (0, 1)}]

    assert actual == expected

    index_list = [12]
    actual = separate_into_chunks(
                index_list=index_list,
                chunk_shape=[5,])

    expected = [{'chunk_spec': ((10, 15),),
                 'output_loc': (0, 1),
                 'chunk_loc': (2, 3)}]

    assert actual == expected
