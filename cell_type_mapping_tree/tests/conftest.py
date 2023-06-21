import pytest


@pytest.fixture(scope='session')
def torch_available_for_testing():
    try:
        import torch
        return True
    except ImportError:
        return False
