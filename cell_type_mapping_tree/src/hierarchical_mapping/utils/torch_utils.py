import os

def is_torch_available():
    """
    Is torch available
    """
    if not hasattr(is_torch_available, 'value'):
        value = False
        try:
            import torch
            value = True
        except:
            pass
        is_torch_available.value = value
    return is_torch_available.value


def is_cuda_available():
   """
   Is cuda available
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
    env_var = 'AIBS_BKP_USE_TORCH'
    if env_var in os.environ:
        if os.environ[env_var] == 'false':
            return False

    if not is_torch_available():
        return False

    if is_cuda_available():
        return True

    if env_var not in os.environ:
        return False

    if os.environ[env_var] == 'true':
        return True

    return False
