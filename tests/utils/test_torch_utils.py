import pytest

import itertools
import unittest.mock
import warnings

import cell_type_mapper.utils.torch_utils as torch_utils


@pytest.mark.parametrize(
    "is_cuda, is_torch",
    itertools.product((True, False), (True, False))
)
def test_use_torch(is_cuda, is_torch):

    def return_false():
        return False

    def return_true():
        return True

    is_cuda_fn = 'cell_type_mapper.utils.torch_utils.is_cuda_available'
    if is_cuda:
        is_cuda_mock = return_true
    else:
        is_cuda_mock = return_false

    is_torch_fn = 'cell_type_mapper.utils.torch_utils.is_torch_available'
    if is_torch:
        is_torch_mock = return_true
    else:
        is_torch_mock = return_false

    warning_msg = "proceed with the CPU implementation"

    with unittest.mock.patch(is_torch_fn, is_torch_mock):
        with unittest.mock.patch(is_cuda_fn, is_cuda_mock):

            if is_cuda and is_torch:
                # We are now returning False use_torch() regardless
                with pytest.warns(torch_utils.TorchOverrideWarning,
                                  match=warning_msg):
                    assert not torch_utils.use_torch()

                # make sure warning was only emitted on first invocation
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    torch_utils.use_torch()
            else:
                assert not torch_utils.use_torch()
