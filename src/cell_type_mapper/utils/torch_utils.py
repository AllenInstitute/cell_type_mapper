import warnings


def is_torch_available():
    """
    Is torch available
    """
    if not hasattr(is_torch_available, 'value'):
        value = False
        try:
            import torch  # noqa: F401
            value = True
        except ImportError:
            pass
        is_torch_available.value = value
    return is_torch_available.value


def is_cuda_available():
    """
    Is cuda available?
    """
    if not is_torch_available():
        return False
    if not hasattr(is_cuda_available, 'value'):
        import torch
        is_cuda_available.value = torch.cuda.is_available()
    return is_cuda_available.value


def find_num_gpus():
    if not is_torch_available():
        return 0
    if not hasattr(find_num_gpus, 'value'):
        import torch
        find_num_gpus.value = torch.cuda.device_count()
    return find_num_gpus.value


def use_torch():
    if not is_torch_available():
        return False

    if is_cuda_available():
        return _override_use_torch()

    return False


def _override_use_torch():
    """
    Emit a warning explaining that there is no point in using
    a GPU for this code. Return False for the result of use_torch()
    """
    if not hasattr(_override_use_torch, 'has_warned'):
        _override_use_torch.has_warned = False

    if not _override_use_torch.has_warned:
        msg = (
            "Nominally, your system is configured to use the GPU "
            "implementation of cell_type_mapper. We have found that "
            "the speed-up due to the GPU is not enough to justify "
            "the more stringent memory requirements, so we are no "
            "longer supporting running the cell_type_mapper on a GPU. "
            "Mapping will proceed with the CPU implementation."
        )
        warnings.warn(msg, category=TorchOverrideWarning)
        _override_use_torch.has_warned = True

    return False


class TorchOverrideWarning(UserWarning):
    pass
