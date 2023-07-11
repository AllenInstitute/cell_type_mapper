import pytest

import pathlib

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.taxonomy.data_release_utils import (
    get_header_map,
    get_term_set_map)


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


def test_get_term_set_map(
        tmp_dir_fixture):

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    expected = {
        'a': 'b',
        'c': 'd',
        'e': 'f'
    }

    with open(csv_path, 'w') as dst:
        dst.write('garbage,cluster_annotation_term_set_name,garbage,'
                  'cluster_annotation_term_set_label,garbage\n')
        for k in expected:
            v = expected[k]
            dst.write(f'000,{v},111,{k},2222\n')

    actual = get_term_set_map(csv_path)
    assert actual == expected

    # check if there are multiple identical entries
    with open(csv_path, 'w') as dst:
        dst.write('garbage,cluster_annotation_term_set_name,garbage,'
                  'cluster_annotation_term_set_label,garbage\n')
        for k in expected:
            v = expected[k]
            dst.write(f'000,{v},111,{k},2222\n')
        for k in expected:
            v = expected[k]
            dst.write(f'000,{v},111,{k},2222\n')

    actual = get_term_set_map(csv_path)
    assert actual == expected

    # test if there are multiple conflicting entries
    with open(csv_path, 'a') as dst:
        dst.write('0,x,1,a,3\n')
    with pytest.raises(RuntimeError, match='maps to at least two names'):
        get_term_set_map(csv_path)
