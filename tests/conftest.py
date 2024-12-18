import pytest

import multiprocessing

from cell_type_mapper.utils.utils import (
    _clean_up)

multiprocessing.set_start_method('fork', force=True)


@pytest.fixture(scope='session')
def tmp_dir_fixture(
        tmp_path_factory):
    result = tmp_path_factory.mktemp('cell_type_mapper_')
    yield result
    _clean_up(result)
