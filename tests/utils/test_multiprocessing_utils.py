import pytest

import multiprocessing

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list,
    winnow_process_dict)


def successful_fn(x):
    return

def unsuccessful_fn(x):
    raise RuntimeError("oh no")


def test_winnow_process_list():
    process_list = []
    for ii in range(3):
        p = multiprocessing.Process(
                target=successful_fn,
                args=(2,))
        p.start()
        process_list.append(p)
    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)


    with pytest.raises(RuntimeError, match="One of the processes"):
        process_list = []
        for ii in range(3):
            p = multiprocessing.Process(
                    target=successful_fn,
                   args=(2,))
            p.start()
            process_list.append(p)

        p = multiprocessing.Process(
                target=unsuccessful_fn,
                args=(2,))
        p.start()
        process_list.append(p)
        while len(process_list) > 0:
            process_list = winnow_process_list(process_list)


def test_winnow_process_dict():
    process_dict = dict()
    for ii in range(3):
        p = multiprocessing.Process(
                target=successful_fn,
                args=(2,))
        p.start()
        process_dict[ii] = p
    while len(process_dict) > 0:
        process_dict = winnow_process_dict(process_dict)

    with pytest.raises(RuntimeError, match="One of the processes"):
        process_dict = dict()
        for ii in range(3):
            p = multiprocessing.Process(
                    target=successful_fn,
                   args=(2,))
            p.start()
            process_dict[ii] =p

        p = multiprocessing.Process(
                target=unsuccessful_fn,
                args=(2,))
        p.start()
        process_dict[4] = p
        while len(process_dict) > 0:
            process_dict = winnow_process_dict(process_dict)
