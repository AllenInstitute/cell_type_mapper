import pytest
import zarr
import numpy as np
import pathlib

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.zarr_wrappers import (
    ZarrCacheWrapper,
    separate_into_chunks)


@pytest.fixture(scope='session')
def zarr_path_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('zarr_data'))
    zarr_path = tmp_dir / 'eg.zarr'
    rng = np.random.default_rng(3442123)
    n = 237
    with zarr.open(zarr_path ,'w') as out_file:
        out_file.create(
                name='data',
                shape=(n),
                chunks=30,
                compressor=None,
                dtype=np.float32)

        out_file['data'][:] = rng.random(n)

    yield zarr_path

    _clean_up(tmp_dir)



@pytest.mark.parametrize(
        'i0, i1',
        ((0, 15),
         (25, 43),
         (25, 72),
         (180, 237),
         (222, 237)))
def test_zarr_cache_wrapper(
        zarr_path_fixture,
        i0,
        i1):

    wrapper = ZarrCacheWrapper(zarr_path_fixture / 'data')
    actual = wrapper[i0:i1]

    with zarr.open(zarr_path_fixture, 'r') as baseline_zarr:
        expected = baseline_zarr['data'][i0:i1]

    np.testing.assert_allclose(actual, expected)


def test_separate_into_chunks_qc():
    """
    Test that error is raised when index_list is not contiguous
    """
    index_list = [1,2,3,7]
    with pytest.raises(RuntimeError, match="not contiguous"):
        separate_into_chunks(index_list, 100)

    index_list = [6,7,8,9,8]
    with pytest.raises(RuntimeError, match="not ascending"):
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
