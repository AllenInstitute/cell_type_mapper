class DummyLock(object):

    def __enter__(self):
        pass

    def __exit__(
            self,
            exception_type,
            exception_value,
            exception_traceback):
        pass


def winnow_process_list(
        process_list):
    """
    Loop over a list of processes, popping out any that have
    been completed. Return the winnowed list of processes.
    Parameters
    ----------
    process_list: List[multiprocessing.Process]
    Returns
    -------
    process_list: List[multiprocessing.Process]
    """
    to_pop = []
    for ii in range(len(process_list)-1, -1, -1):
        if process_list[ii].exitcode is not None:
            to_pop.append(ii)
            if process_list[ii].exitcode != 0:
                raise RuntimeError(
                    "One of the processes exited with code "
                    f"{process_list[ii].exitcode}")
    for ii in to_pop:
        process_list.pop(ii)
    return process_list


def winnow_process_dict(
        process_dict):
    """
    Loop over a dict of processes, popping out any that have
    been completed. Return the winnowed dict of processes.
    """
    key_list = list(process_dict.keys())
    for k in key_list:
        if process_dict[k].exitcode is not None:
            if process_dict[k].exitcode != 0:
                raise RuntimeError(
                    f"One of the processes (key={k}) exited with code "
                    f"{process_dict[k].exitcode}")
            process_dict.pop(k)
    return process_dict
