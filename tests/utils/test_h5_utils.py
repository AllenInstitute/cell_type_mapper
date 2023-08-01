import pytest

import h5py
import numpy as np
import pathlib

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.utils.h5_utils import (
    copy_h5_excluding_data)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):

    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('h5_utils'))

    yield tmp_dir

    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def data_fixture():
    rng = np.random.default_rng(876123)

    result = dict()
    result['a'] = rng.random(17)
    result['b'] = 'thequickbrownfox'.encode('utf-8')
    result['c/d'] = rng.random(62)
    result['c/e'] = 'jumpedoverthelazydog'.encode('utf-8')
    result['f/g/h'] = 'youknowhatIamsaing'.encode('utf-8')
    result['f/g/i'] = rng.random(7)
    result['f/j/k'] = rng.random(19)
    result['f/j/l'] = 'seeyoulater'.encode('utf-8')

    return result


@pytest.fixture(scope='module')
def h5_fixture(
        tmp_dir_fixture,
        data_fixture):
    h5_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='baseline_',
        suffix='.h5')
    with h5py.File(h5_path, 'w') as dst:
        for name in data_fixture:
            if name == 'c/d':
                chunks = (12,)
            else:
                chunks = None
            dst.create_dataset(
                name,
                data=data_fixture[name],
                chunks=chunks)

    return h5_path


@pytest.mark.parametrize(
    'excluded_groups,excluded_datasets',
    [
     (None, None),
     (None, ['b']),
     (None, ['f/g/i']),
     (None, ['b', 'f/g/i']),
     (['f'], None),
     (['c', 'f/j'], None),
     (['c', 'f/j'], ['b'])
    ])
def test_h5_copy_util(
        tmp_dir_fixture,
        data_fixture,
        h5_fixture,
        excluded_groups,
        excluded_datasets):

    dst_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        prefix='dst_file_',
        suffix='.h5')

    copy_h5_excluding_data(
        src_path=h5_fixture,
        dst_path=dst_path,
        excluded_datasets=excluded_datasets,
        excluded_groups=excluded_groups)

    # make sure src file was unchanged
    with h5py.File(h5_fixture, 'r') as src:
        for name in data_fixture:
            if isinstance(data_fixture[name], np.ndarray):
                np.testing.assert_allclose(
                    data_fixture[name],
                    src[name][()],
                    atol=0.0,
                    rtol=1.0e-6)
            else:
                assert data_fixture[name] == src[name][()]

    to_exclude =[]
    if excluded_datasets is not None:
        to_exclude += list(excluded_datasets)
    if excluded_groups is not None:
        to_exclude += list(excluded_groups)

    # test contents of dst file
    with h5py.File(dst_path, 'r') as src:
        for name in data_fixture:
            is_excluded = False
            for bad in to_exclude:
                if bad in name:
                    is_excluded = True

            if is_excluded:
                assert name not in src
            else:
               if isinstance(data_fixture[name], np.ndarray):
                   np.testing.assert_allclose(
                       data_fixture[name],
                       src[name][()],
                       atol=0.0,
                       rtol=1.0e-6)
               else:
                   assert data_fixture[name] == src[name][()]
