from typing import Union, List, Tuple, Optional, Any
import datetime
import numpy as np
import os
import pathlib
import tempfile
import time


def _clean_up(target_path):
    if target_path is None:
        return
    target_path = pathlib.Path(target_path)
    if target_path.is_file():
        target_path.unlink()
    elif target_path.is_dir():
        for sub_path in target_path.iterdir():
            _clean_up(sub_path)
        target_path.rmdir()


def file_size_in_bytes(file_path, chunk_size=1000000000):
    n_bytes = 0
    with open(file_path, 'rb') as in_file:
        chunk = in_file.read(chunk_size)
        n_bytes += len(chunk)
    return n_bytes


def merge_index_list(
        index_list: Union[list, np.ndarray]) -> List[Tuple[int, int]]:
    """
    Take a list of integers, merge those that can be merged into
    (min, max) ranges for slicing array. Return as a list of those
    tuples. Note that max will be 1 greater than any value in the array
    because of the way array slicing works.
    """
    index_list = np.unique(index_list)
    diff_list = np.diff(index_list)
    breaks = np.where(diff_list > 1)[0]
    result = []
    min_dex = 0
    for max_dex in breaks:
        result.append((index_list[min_dex],
                       index_list[max_dex]+1))
        min_dex = max_dex+1
    result.append((index_list[min_dex], index_list[-1]+1))
    return result


def print_timing(
        t0: float,
        i_chunk: int,
        tot_chunks: int,
        unit: str = 'min',
        nametag: Optional[Any] = None,
        msg: Optional[str] = None):

    if unit not in ('sec', 'min', 'hr'):
        raise RuntimeError(f"timing unit {unit} nonsensical")

    denom = {'min': 60.0,
             'hr': 3600.0,
             'sec': 1.0}[unit]

    duration = (time.time()-t0)/denom
    per = duration/max(1, i_chunk)
    pred = per*tot_chunks
    remain = pred-duration
    this_msg = f"{i_chunk} of {tot_chunks} in {duration:.2e} {unit}; "
    this_msg += f"predict {remain:.2e} {unit} of {pred:.2e} {unit} left"
    if nametag is not None:
        this_msg = f"{nametag} -- {msg}"

    if msg is not None:
        this_msg = f"{this_msg} -- {msg}"

    print(this_msg)


def mkstemp_clean(
        dir: Optional[Union[pathlib.Path, str]] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        delete=False) -> str:
    """
    A thin wrapper around tempfile mkstemp that automatically
    closes the file descripter returned by mkstemp.

    Parameters
    ----------
    dir: Optional[Union[pathlib.Path, str]]
        The directory where the tempfile is created

    prefix: Optional[str]
        The prefix of the tempfile's name

    suffix: Optional[str]
        The suffix of the tempfile's name

    delete:
        if True, delete the file
        (dangerous, as could interfere with tempfile.mkstemp's
        ability to create unique file names; should only be used
        during testing where it is important that the tempfile
        doesn't actually exist)

    Returns
    -------
    file_path: str
        Path to a valid temporary file

    Notes
    -----
    Because this calls tempfile mkstemp, the file will be created,
    though it will be empty. This wrapper is needed because
    mkstemp automatically returns an open file descriptor, which was
    been causing some of our unit tests to overwhelm the OS's limit
    on the number of open files.
    """
    (descriptor,
     file_path) = tempfile.mkstemp(
                     dir=dir,
                     prefix=prefix,
                     suffix=suffix)

    os.close(descriptor)
    if delete:
        os.unlink(file_path)
    return file_path


def get_timestamp():
    """
    Return a string with the current timestamp
    """
    now = datetime.datetime.now()
    result = f"{now.year:04d}-{now.month:02d}"
    result += f"-{now.day:02d}-{now.hour:02d}"
    result += f"-{now.minute:02d}-{now.second:02d}"
    return result


def update_timer(name, t, timers=None):
    if timers is not None:
        if timers.get(name) is not None:
            timers.get(name).update(time.time() - t)


def choose_int_dtype(
        x_minmax):
    """
    Parameters
    ----------
    x_minmax:
        Tuple of minimum and maximum values

    Returns
    -------
    smallest int dtype that can accommodate that range
    """
    output_dtype = None
    int_min = np.round(x_minmax[0])
    int_max = np.round(x_minmax[1])

    for candidate in (np.uint8, np.int8, np.uint16, np.int16,
                      np.uint32, np.int32, np.uint64, np.int64):
        this_info = np.iinfo(candidate)
        if int_min >= this_info.min and int_max <= this_info.max:
            output_dtype = candidate
            break
    if output_dtype is None:
        output_dtype = int
    return output_dtype


def clean_for_json(data):
    """
    Iteratively walk through data, converting np.int64 to int as needed

    Also convert sets into sorted lists and np.ndarrays into lists
    """
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, list) or isinstance(data, tuple):
        return [clean_for_json(el) for el in data]
    elif isinstance(data, set):
        new_data = list(data)
        new_data.sort()
        return clean_for_json(new_data)
    elif isinstance(data, np.ndarray):
        return clean_for_json(data.tolist())
    elif isinstance(data, dict):
        new_data = {
            key: clean_for_json(data[key])
            for key in data
        }
        return new_data
    return data


def clean_for_uns_serialization(data):
    """
    anndata's uns object will treat any dict keys with a '/' in them
    as though they are separate levels in the HDF5 filesystem. This
    function is part of a workaround to that behavior. It scans through
    data and replaces any '/' in keys with '$'

    Parameters
    ----------
    data:
        Any blob being serialized

    Returns
    -------
    data_cleaned:
        '/' replaced with '$'
    """
    return _clean_for_uns(
                data=data,
                old_symbol='/',
                new_symbol='$')


def clean_for_uns_deserialization(data):
    """
    anndata's uns object will treat any dict keys with a '/' in them
    as though they are separate levels in the HDF5 filesystem. This
    function is part of a workaround to that behavior. It scans through
    data and replaces any '$' in keys with '/'

    Parameters
    ----------
    data:
        Any blob being serialized

    Returns
    -------
    data_cleaned:
        '$' replaced with '/'
    """
    return _clean_for_uns(
                data=data,
                old_symbol='$',
                new_symbol='/')


def _clean_for_uns(
        data,
        old_symbol,
        new_symbol):
    """
    Parameters
    ----------
    data:
        blob being serialized
    old_symbol:
        a str
    new_symbol:
        a str

    Returns
    -------
    cleaned_data:
        data, but keys in dicts are modified so that
        old_symbol is replaced with new_symbol
    """
    if isinstance(data, dict):
        cleaned_data = dict()
        for key in data:
            if isinstance(key, str):
                new_key = key.replace(old_symbol, new_symbol)
            else:
                new_key = key
            cleaned_data[new_key] = _clean_for_uns(
                                         data=data[key],
                                         old_symbol=old_symbol,
                                         new_symbol=new_symbol)
    elif isinstance(data, list):
        cleaned_data = [
            _clean_for_uns(
                data=element,
                old_symbol=old_symbol,
                new_symbol=new_symbol)
            for element in data]
    else:
        cleaned_data = data

    return cleaned_data
