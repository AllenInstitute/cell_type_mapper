import pytest

import pathlib

from hierarchical_mapping.utils.utils import (
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.taxonomy.data_release_utils import (
    get_header_map)


@pytest.fixture(scope='module')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('taxonomy_data_release_'))
    yield tmp_dir
    _clean_up(tmp_dir)


def test_get_header_map(
    tmp_dir_fixture):

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')
    with open(csv_path, 'w') as out_file:
        out_file.write('a,b,c,d,e,f,e,g\n')

    actual = get_header_map(
        csv_path=csv_path,
        desired_columns = ['b', 'f', 'c'])

    expected = {'b': 1, 'c': 2, 'f': 5}
    assert expected == actual

    with pytest.raises(RuntimeError, match="could not find column 'x'"):
        get_header_map(
            csv_path=csv_path,
            desired_columns=['a', 'b', 'x'])

    with open(csv_path, 'w') as out_file:
        out_file.write('a,b,c,d,e,b,f,g\n')

    with pytest.raises(RuntimeError, match="'b' occurs more than once"):
        get_header_map(
            csv_path=csv_path,
            desired_columns=['a', 'b', 'c'])
