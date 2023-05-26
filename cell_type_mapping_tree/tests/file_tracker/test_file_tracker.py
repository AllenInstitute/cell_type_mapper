import pytest

import pathlib

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.file_tracker.file_tracker import (
    FileTracker)

@pytest.fixture
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('file_tracker'))
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture
def actual_file(tmp_dir_fixture):
    fpath = mkstemp_clean(dir=tmp_dir_fixture, suffix='.txt')
    with open(fpath, 'w') as out_file:
        out_file.write('hello')
    return str(pathlib.Path(fpath).resolve().absolute())

@pytest.fixture
def actual_file_bytes(actual_file):
    with open(actual_file, 'rb') as in_file:
        data = in_file.read()
    return data


@pytest.mark.parametrize('input_only', [True, False])
def test_input_file(
        tmp_dir_fixture,
        actual_file,
        actual_file_bytes,
        input_only):
    """
    Test that, when tracking a file that already exists,
    it is not changed (but is faithfully copied)
    """
    tracker = FileTracker(tmp_dir=tmp_dir_fixture)
    this_dir = str(tracker.tmp_dir.resolve().absolute())
    assert this_dir != str(tmp_dir_fixture.resolve().absolute())

    tracker.add_file(
        actual_file,
        input_only=input_only)

    assert actual_file != tracker.real_location(actual_file)

    tmp_path = pathlib.Path(tracker.real_location(actual_file))

    with open(tracker.real_location(actual_file), 'rb') as in_file:
        these_bytes = in_file.read()
    assert these_bytes == actual_file_bytes

    assert tmp_path.is_file()

    # now change it; make sure it was not copied
    # back on deletion it already existed
    with open(tmp_path, 'w') as out_file:
        out_file.write('different')

    with open(tracker.real_location(actual_file), 'rb') as in_file:
        these_bytes = in_file.read()
    assert these_bytes != actual_file_bytes

    del tracker
    assert not tmp_path.is_file()

    with open(actual_file, 'rb') as in_file:
        these_bytes = in_file.read()
    assert these_bytes == actual_file_bytes


def test_error_if_input_does_not_exit(
        tmp_dir_fixture):
    """
    Test that an error is raised if an input file does
    not exist
    """
    tracker = FileTracker(tmp_dir=tmp_dir_fixture)
    with pytest.raises(RuntimeError, match="is not a file"):
        tracker.add_file('garbage.txt', input_only=True)


def test_creating_new_file(
        tmp_dir_fixture):
    """
    Test that a file is created and copied to the correct
    location once the tracker is deleted.
    """

    final_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir_fixture,
                suffix='.txt'))

    final_path.unlink()

    tracker = FileTracker(
        tmp_dir=tmp_dir_fixture)

    tracker.add_file(
        final_path,
        input_only=False)

    assert not final_path.is_file()
    with open(tracker.real_location(final_path), 'w') as out_file:
        out_file.write('silly')
    with open(tracker.real_location(final_path), 'rb') as in_file:
        expected_bytes = in_file.read()

    assert not final_path.is_file()

    del tracker

    assert final_path.is_file()
    with open(final_path, 'rb') as in_file:
        actual_bytes = in_file.read()
    assert actual_bytes == expected_bytes
