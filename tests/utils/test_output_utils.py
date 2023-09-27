import pytest

import numpy as np
import pathlib

from cell_type_mapper.utils.utils import (
    _clean_up,
    mkstemp_clean)

from cell_type_mapper.utils.output_utils import (
    blob_to_df,
    blob_to_csv)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('output'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='module')
def results_fixture():
    results = [
        {'cell_id': 'a',
         'level1': {'assignment': 'alice',
                    'confidence': 0.01234567,
                    'corr': 0.112253,
                    'runners_up': ['a', 'b', 'd']},
         'level3': {'assignment': 'bob',
                    'confidence': 0.2,
                    'corr': 0.4,
                    'runners_up': ['c']},
         'level7': {'assignment': 'cheryl',
                    'confidence': 0.245,
                    'corr': 0.33332}
        },
        {'cell_id': 'b',
         'level1': {'assignment': 'roger',
                    'confidence': 0.11119,
                    'corr': 0.1},
         'level3': {'assignment': 'dodger',
                    'confidence': 0.3,
                    'corr': 0.9,
                    'runners_up': ['a', 'f', 'b', 'b']},
         'level7': {'assignment': 'brooklyn',
                    'confidence': 0.5,
                    'corr': 0.11723}
        }
    ]
    return results


def test_blob_to_df(results_fixture):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']
        @property
        def leaf_level(self):
            return 'level7'
        def label_to_name(self, level, label, name_key='gar'):
            return label
        def level_to_name(self, level_label):
            return level_label

    actual_df = blob_to_df(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree())

    assert len(actual_df) == 2

    expected_columns = set([
        "cell_id",
        "level1_name",
        "level1_label",
        "level1_confidence",
        "level1_corr",
        "level1_runners_up_0",
        "level1_runners_up_1",
        "level1_runners_up_2",
        "level3_name",
        "level3_label",
        "level3_confidence",
        "level3_corr",
        "level3_runners_up_0",
        "level3_runners_up_1",
        "level3_runners_up_2",
        "level3_runners_up_3",
        "level7_name",
        "level7_label",
        "level7_alias",
        "level7_confidence",
        "level7_corr"])

    assert set(actual_df.columns) == expected_columns

    for record in results_fixture:
        sub_df = actual_df[actual_df['cell_id']==record['cell_id']]
        assert len(sub_df) == 1
        for level in ('level1', 'level3', 'level7'):
            for k in ('name', 'label'):
                assert sub_df[f'{level}_{k}'].values == record[level]['assignment']
            if level == 'level7':
                assert sub_df[f'{level}_alias'].values == record[level]['assignment']
            for k in ('confidence', 'corr'):
                np.testing.assert_allclose(
                    sub_df[f'{level}_{k}'],
                    record[level][k])

            if level == 'level7':
                continue
            if 'runners_up' in record[level]:
                expected_runners_up = record[level]['runners_up']
            else:
                expected_runners_up = []
            for idx in range(len(expected_runners_up)):
                assert sub_df[f'{level}_runners_up_{idx}'].values == expected_runners_up[idx]

            if level == 'level1' and len(expected_runners_up) < 3:
                for idx in range(len(expected_runners_up), 3):
                    sub_df[f'{level}_runners_up_{idx}'].values is None
            elif level == 'level3' and len(expected_runners_up) < 4:
                for idx in range(len(expected_runners_up), 4):
                    sub_df[f'{level}_runners_up_{idx}'] is None


@pytest.mark.parametrize('with_metadata', [True, False])
def test_blob_to_csv(tmp_dir_fixture, with_metadata, results_fixture):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']
        @property
        def leaf_level(self):
            return 'level7'
        def label_to_name(self, level, label, name_key='gar'):
            return label
        def level_to_name(self, level_label):
            return level_label

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    if with_metadata:
        metadata_path = 'path/to/my/cool_file.txt'
        n_expected = 6
        n_offset = 1
    else:
        metadata_path = None
        n_expected = 5
        n_offset = 0

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    header_line = ('cell_id,level1_label,level1_name,level1_confidence,'
                   'level3_label,level3_name,level3_confidence,level7_label,'
                   'level7_name,level7_alias,level7_confidence\n')
    assert actual_lines[2+n_offset] == header_line

    cell0 = 'a,alice,alice,0.0123,bob,bob,0.2000,cheryl,cheryl,cheryl,0.2450\n'
    assert actual_lines[3+n_offset] == cell0

    cell1 = 'b,roger,roger,0.1112,dodger,dodger,0.3000,brooklyn,brooklyn,brooklyn,0.5000\n'
    assert actual_lines[4+n_offset] == cell1

    # test again using a different confidence stat

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path,
        confidence_key='corr',
        confidence_label='number')

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    header_line = ('cell_id,level1_label,level1_name,level1_number,'
                   'level3_label,level3_name,level3_number,level7_label,'
                   'level7_name,level7_alias,level7_number\n')
    assert actual_lines[2+n_offset] == header_line

    cell0 = 'a,alice,alice,0.1123,bob,bob,0.4000,cheryl,cheryl,cheryl,0.3333\n'
    assert actual_lines[3+n_offset] == cell0

    cell1 = 'b,roger,roger,0.1000,dodger,dodger,0.9000,brooklyn,brooklyn,brooklyn,0.1172\n'
    assert actual_lines[4+n_offset] == cell1



@pytest.mark.parametrize('with_metadata', [True, False])
def test_blob_to_csv_with_mapping(tmp_dir_fixture, with_metadata, results_fixture):
    """
    Test with a name mapping
    """

    name_mapper = {
        'level1': {
            'alice': {
                'name': 'beverly'
            },
            'roger': {
                'name': 'jane'
            }
        },
        'level3': {
            'bob': {
                'name': 'X'
            },
            'dodger': {
                'name': 'Y'
            }
        },
        'level7': {
            'cheryl': {
                'name': 'tom',
                'alias': '77'
            },
            'brooklyn': {
                'name': 'cleveland',
                'alias': '88'
            }
        }
    }

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']
        @property
        def leaf_level(self):
            return 'level7'
        def label_to_name(self, level, label, name_key='gar'):
            return name_mapper[level][label][name_key]
        def level_to_name(self, level_label):
            return level_label

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    if with_metadata:
        metadata_path = 'path/to/my/cool_file.txt'
        n_expected = 6
        n_offset = 1
    else:
        metadata_path = None
        n_expected = 5
        n_offset = 0

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    header_line = ('cell_id,level1_label,level1_name,level1_confidence,'
                   'level3_label,level3_name,level3_confidence,level7_label,'
                   'level7_name,level7_alias,level7_confidence\n')
    assert actual_lines[2+n_offset] == header_line

    cell0 = 'a,alice,beverly,0.0123,bob,X,0.2000,cheryl,tom,77,0.2450\n'
    assert actual_lines[3+n_offset] == cell0

    cell1 = 'b,roger,jane,0.1112,dodger,Y,0.3000,brooklyn,cleveland,88,0.5000\n'
    assert actual_lines[4+n_offset] == cell1

    # test again using a different confidence stat

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path,
        confidence_key='corr',
        confidence_label='number')

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    header_line = ('cell_id,level1_label,level1_name,level1_number,'
                   'level3_label,level3_name,level3_number,level7_label,'
                   'level7_name,level7_alias,level7_number\n')
    assert actual_lines[2+n_offset] == header_line

    cell0 = 'a,alice,beverly,0.1123,bob,X,0.4000,cheryl,tom,77,0.3333\n'
    assert actual_lines[3+n_offset] == cell0

    cell1 = 'b,roger,jane,0.1000,dodger,Y,0.9000,brooklyn,cleveland,88,0.1172\n'
    assert actual_lines[4+n_offset] == cell1


@pytest.mark.parametrize('with_metadata', [True, False])
def test_blob_to_csv_level_map(tmp_dir_fixture, with_metadata, results_fixture):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']
        @property
        def leaf_level(self):
            return 'level7'
        def label_to_name(self, level, label, name_key='gar'):
            return label
        def level_to_name(self, level_label):
            n = level_label.replace('level', '')
            n = int(n)**2
            return f'salted_{n}'

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    if with_metadata:
        metadata_path = 'path/to/my/cool_file.txt'
        n_expected = 7
        n_offset = 1
    else:
        metadata_path = None
        n_expected = 6
        n_offset = 0

    blob_to_csv(
        results_blob=results_fixture,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=metadata_path)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == n_expected

    if with_metadata:
        metadata_line = '# metadata = cool_file.txt\n'
        assert actual_lines[0] == metadata_line

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0+n_offset] == taxonomy_line

    readable_taxonomy_line = ('# readable taxonomy hierarchy = '
                              '["salted_1", '
                              '"salted_9", '
                              '"salted_49"]\n')
    assert actual_lines[1+n_offset] == readable_taxonomy_line

    header_line = ('cell_id,'
                   'salted_1_label,salted_1_name,salted_1_confidence,'
                   'salted_9_label,salted_9_name,salted_9_confidence,'
                   'salted_49_label,salted_49_name,salted_49_alias,'
                   'salted_49_confidence\n')
    # version line is at 2+n_offset
    assert actual_lines[3+n_offset] == header_line
