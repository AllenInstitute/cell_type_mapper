import pytest

import pathlib

from hierarchical_mapping.utils.utils import (
    _clean_up,
    mkstemp_clean)

from hierarchical_mapping.utils.output_utils import (
    blob_to_csv)


@pytest.fixture
def tmp_dir_fixture(tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('output'))
    yield tmp_dir
    _clean_up(tmp_dir)


def test_blob_to_csv(tmp_dir_fixture):

    class DummyTree(object):
        hierarchy = ['level1', 'level3', 'level7']

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

    blob_to_csv(
        results_blob=results,
        taxonomy_tree=DummyTree(),
        output_path=csv_path,
        metadata_path=None)

    with open(csv_path, 'r') as src:
        actual_lines = src.readlines()
    assert len(actual_lines) == 4

    taxonomy_line = '# taxonomy hierarchy = ["level1", "level3", "level7"]\n'
    assert actual_lines[0] == taxonomy_line
    header_line = 'cell_id,level1,level1_confidence,level3,level3_confidence,'
    header_line += 'level7,level7_confidence\n'
    assert actual_lines[1] == header_line

    cell0 = 'a,alice,0.0123,bob,0.2000,cheryl,0.2450\n'
    assert actual_lines[2] == cell0
    cell1 = 'b,roger,0.1112,dodger,0.3000,brooklyn,0.5000\n'
    assert actual_lines[3] == cell1
