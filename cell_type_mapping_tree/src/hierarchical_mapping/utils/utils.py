from typing import Union, List, Tuple
import numpy as np
import pathlib
import time


def _clean_up(target_path):
    target_path = pathlib.Path(target_path)
    if target_path.is_file():
        target_path.unlink()
    elif target_path.is_dir():
        for sub_path in target_path.iterdir():
            _clean_up(sub_path)
        target_path.rmdir()


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


def refactor_row_chunk_list(
        row_chunk_list,
        final_chunk_size):
    """
    Rearrange row_chunk_list so that each chunk
    is an integer multiple of final_chunk_size.

    This will allow us to parallelize the work of rearranging
    the anndata file such that no two workers are touching
    the same chunk.
    """

    output_chunk_list = []
    remainder = None
    for raw_input_chunk in row_chunk_list:
        if remainder is not None:
            input_chunk = remainder + raw_input_chunk
        else:
            input_chunk = raw_input_chunk

        if len(input_chunk) % final_chunk_size == 0:
            output_chunk_list.append(input_chunk)
            remainder = None
        else:
            factor = len(input_chunk) // final_chunk_size
            n = factor*final_chunk_size
            if n > 0:
                output_chunk_list.append(input_chunk[:n])
            remainder = input_chunk[n:]

    if remainder is not None:
        output_chunk_list.append(remainder)

    return output_chunk_list
