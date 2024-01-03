import pathlib
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean)

from cell_type_mapper.utils.cloud_utils import (
    is_exposed,
    sanitize_paths)



def test_is_exposed(tmp_dir_fixture):
    assert not is_exposed(pathlib.Path('.'))
    assert not is_exposed(pathlib.Path('/'))

    # nonsense path
    garbage_path = pathlib.Path('/this/is/silly')
    assert not garbage_path.exists()
    assert not is_exposed(garbage_path)

    # file that exists
    other_path = pathlib.Path(
        mkstemp_clean(dir=tmp_dir_fixture, suffix='.csv'))
    assert other_path.is_file()
    assert is_exposed(other_path)

    # file that does not exist but has valid parent
    other_path = pathlib.Path(
        mkstemp_clean(dir=tmp_dir_fixture, suffix='.csv'))
    other_path.unlink()
    assert not other_path.exists()
    assert is_exposed(other_path)

    # dir that exists
    dir_path = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture))
    assert dir_path.is_dir()
    assert is_exposed(dir_path)

    # dir that does not exist but has valid parent
    dir_path = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir_fixture))
    dir_path.rmdir()
    assert not dir_path.is_dir()
    assert is_exposed(dir_path)


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
