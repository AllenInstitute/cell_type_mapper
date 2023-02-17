from typing import Union, List, Tuple
import numpy as np
import time


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
    breaks = np.where(diff_list>1)[0]
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
        unit: str = 'min'):

    if unit not in ('sec', 'min', 'hr'):
        raise RuntimeError(f"timing unit {unit} nonsensical")

    denom = {'min': 60.0,
             'hr': 3600.0,
             'sec': 1.0}[unit]

    duration = (time.time()-t0)/denom
    per = duration/i_chunk
    pred = per*tot_chunks
    remain = pred-duration
    print(f"{i_chunk} of {tot_chunks} in {duration:.2e} {unit}; "
          f"predict {remain:.2e} of {pred:.2e} left")
