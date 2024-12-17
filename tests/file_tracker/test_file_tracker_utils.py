import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean
)

from cell_type_mapper.file_tracker.utils import (
    is_file_under_dir
)


def test_is_file_under_dir(tmp_dir_fixture):

    dir_a = tempfile.mkdtemp(dir=tmp_dir_fixture)
    dir_b = tempfile.mkdtemp(dir=dir_a)
    dir_c = tempfile.mkdtemp(dir=tmp_dir_fixture)

    file_a = mkstemp_clean(dir=dir_b)

    assert is_file_under_dir(file_path=file_a, dir_path=dir_b)
    assert is_file_under_dir(file_path=file_a, dir_path=dir_a)
    assert not is_file_under_dir(file_path=file_a, dir_path=dir_c)
