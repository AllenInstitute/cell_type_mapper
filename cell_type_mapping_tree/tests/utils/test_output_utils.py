import pytest

import pathlib

from hierarchical_mapping.utils.utils import (
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.utils.output_utils import (
    blob_to_csv)


@pytest.fixture(scope='module')
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('output'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.mark.parametrize('with_metadata', [True, False])
def test_blob_to_csv(tmp_dir_fixture, with_metadata):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']
        def label_to_name(self, level, label, name_key='gar'):
            return label

    results = [
        {'cell_id': 'a',
         'level1': {'assignment': 'alice',
                    'confidence': 0.01234567},
         'level3': {'assignment': 'bob',
                    'confidence': 0.2},
         'level7': {'assignment': 'cheryl',
                    'confidence': 0.245}
        
        },
        {'cell_id': 'b',
         'level1': {'assignment': 'roger',
                    'confidence': 0.11119},
         'level3': {'assignment': 'dodger',
                    'confidence': 0.3},
         'level7': {'assignment': 'brooklyn',
                    'confidence': 0.5}
        }
    ]

    csv_path = mkstemp_clean(
        dir=tmp_dir_fixture,
        suffix='.csv')

    if with_metadata:
        metadata_path = 'path/to/my/cool_file.txt'
        n_expected = 5
        n_offset = 1
    else:
        metadata_path = None
        n_expected = 4
        n_offset = 0

    blob_to_csv(
        results_blob=results,
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

    header_line = ('cell_id,level1_label,level1_confidence,'
                   'level3_label,level3_confidence,level7_label,'
                   'level7_confidence\n')
    assert actual_lines[1+n_offset] == header_line

    cell0 = 'a,alice,0.0123,bob,0.2000,cheryl,0.2450\n'
    assert actual_lines[2+n_offset] == cell0

    cell1 = 'b,roger,0.1112,dodger,0.3000,brooklyn,0.5000\n'
    assert actual_lines[3+n_offset] == cell1
