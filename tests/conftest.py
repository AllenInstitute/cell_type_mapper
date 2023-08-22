import pytest

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

from cell_type_mapper.utils.utils import (
    _clean_up)


@pytest.fixture(scope='session')
def tmp_dir_fixture(
        tmp_path_factory):
    result = tmp_path_factory.mktemp('cell_type_mapper_')
    yield result
    _clean_up(result)
