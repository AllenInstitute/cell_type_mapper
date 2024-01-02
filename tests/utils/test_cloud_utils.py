import pathlib
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.cloud_utils import (
    sanitize_paths)


def test_sanitize_paths(tmp_dir_fixture):
    this_tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=tmp_dir_fixture))
    f0 = str((this_tmp_dir/"f0.csv").resolve().absolute())
    f1 = str((this_tmp_dir/"f1.csv").resolve().absolute())
    f2 = str((this_tmp_dir/"f2.csv").resolve().absolute())
    f3 = str((this_tmp_dir/"f3.csv").resolve().absolute())
    f4 = str((this_tmp_dir/"f4.csv").resolve().absolute())
    f5 = str((this_tmp_dir/"f5.csv").resolve().absolute())
    for n in (f0, f1, f2, f3, f4, f5):
        with open(n, 'w') as dst:
            dst.write('garbage')

    other_dir = this_tmp_dir / 'a_dir'
    other_dir.mkdir()

    config = {
        'a': 'b',
        'c': [1, 2, 3],
        'e': f'this is fun {f1}',
        'f': [f2, f3, f'and now {f4}'],
        'g': {
           'aa': f5,
           'bb': f'so what {f1}',
           'cc': 2,
           'dd': str(other_dir)
        }
    }

    actual = sanitize_paths(config)

    expected = {
        'a': 'b',
        'c': [1, 2, 3],
        'e': 'this is fun f1.csv',
        'f': ['f2.csv', 'f3.csv', 'and now f4.csv'],
        'g': {
           'aa': 'f5.csv',
           'bb': 'so what f1.csv',
           'cc': 2,
           'dd': 'a_dir'
        }
    }

    assert actual == expected
    assert actual != config
