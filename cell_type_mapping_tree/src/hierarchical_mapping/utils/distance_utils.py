import numpy as np
import logging
import time

from hierarchical_mapping.utils.utils import (
    update_timer)

TORCH_AVAILABLE = False
LOGGER = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
        LOGGER.info("Found CUDA. Runnig on GPU.")
    LOGGER.info("Torch found.")
except ImportError as e:
    LOGGER.info(f"Torch not found. Running on CPU. {str(e)}")
    pass


def correlation_nearest_neighbors(
        baseline_array,
        query_array,
        return_correlation=False,
        gpu_index=0,
        timers=None):
    """
    Find the index of the nearest neighbors (by correlation
    distance) of the cells in

    Parameters
    ----------
    baseline_array:
        A (n_cells_0, n_genes) np.ndarray. The cell x gene data
        from which nearest neighbors will be drawn
    query_array:
        A (n_cells_1, n_genes) np.ndarray. The cell x gene data
        whose nearest neighbors are desired
    return_correlation:
        If True, also return the correlation values of the best
        fit

    Returns
    -------
    A (n_cells_1, ) np.ndarray of integers representing
    the index of the nearest cell from baseline_arry to each
    cell in query_array (i.e. the returned value at 11 is the
    nearest neighbor of query_array[11, :])
    """
    t = time.time()
    correlation_array = correlation_dot(baseline_array,
                                        query_array,
                                        gpu_index=gpu_index,
                                        timers=timers)
    update_timer("correlation_dot", t, timers)

    if TORCH_AVAILABLE:
        t = time.time()
        max_idx = torch.argmax(correlation_array, dim=0)
        update_timer("argmax", t, timers)

        # t = time.time()
        # max_idx = max_idx.cpu().numpy()
        # update_timer("tocpu", t, timers)

    else:
        t = time.time()
        max_idx = np.argmax(correlation_array, axis=0)
        update_timer("argmax", t, timers)

    if not return_correlation:
        return max_idx

    t = time.time()
    max_val = correlation_array[
        max_idx,
        np.arange(correlation_array.shape[1])]
    update_timer("correlationarraynge", t, timers)

    return max_idx, max_val


def correlation_distance(
        arr0,
        arr1):
    """
    Return the correlation distance between the rows of two
    (n_cells, n_genes) arrays

    Parameters
    ----------
    arr0:
        A (n_cells_0, n_genes) np.ndarray
    arr1:
        A (n_cells_1, n_genes) np.ndarray

    Returns
    -------
    corr_dist:
        A (n_cells_0, n_cells_1) np.ndarray of the correlation
        distances between the rows of arr0, arr1
    """
    return 1.0-correlation_dot(arr0, arr1)


def correlation_dot(arr0,
                    arr1,
                    gpu_index=0,
                    timers=None):
    """
    Return the correlation between the rows of two
    (n_cells, n_genes) arrays

    Parameters
    ----------
    arr0:
        A (n_cells_0, n_genes) np.ndarray
    arr1:
        A (n_cells_1, n_genes) np.ndarray

    Returns
    -------
    corr:
        A (n_cells_0, n_cells_1) np.ndarray of the correlation
        between the rows of arr0, arr1

    Note
    ----
    For any row where the standard deviation is zero (i.e. the row
    has a constant expression value), the correlation will be returned
    as zero, instead of NaN.
    """

    t = time.time()
    arr0 = _subtract_mean_and_normalize(arr0,
                                        do_transpose=False,
                                        gpu_index=gpu_index,
                                        timers=timers)
    update_timer("sn1", t, timers)

    t = time.time()
    arr1 = _subtract_mean_and_normalize(arr1,
                                        do_transpose=True,
                                        gpu_index=gpu_index,
                                        timers=timers)
    update_timer("sn2", t, timers)

    if TORCH_AVAILABLE:

        t = time.time()
        correlation = torch.matmul(arr0, arr1)
        update_timer("matmul", t, timers)

        t = time.time()
        del arr0
        del arr1
        update_timer("dele", t, timers)
        return correlation

    # non-gpu path
    t = time.time()
    correlation = np.dot(arr0, arr1)
    update_timer("matmul", t, timers)

    return correlation


def _subtract_mean_and_normalize(data,
                                 do_transpose=False,
                                 gpu_index=0,
                                 timers=None):
    """
    Prep an array of cell x gene data for correlation distance
    computation.

    Parameters
    ----------
    data:
        A (n_cells, n_genes) np.ndarray
    do_transpose:
        A boolean. If False, return a (n_cells, n_gene)
        array. If True, return a (n_gene, n_cells) array

    Return
    ------
    data with the mean of each row subtracted and the
    each mean-subtracted row normalized to its L2 norm.
    (The array will also be transposed relative to the
    input if do_transpose)
    """
    if TORCH_AVAILABLE:
        return _subtract_mean_and_normalize_gpu(data,
                                                do_transpose=do_transpose,
                                                gpu_index=gpu_index,
                                                timers=timers)
    return _subtract_mean_and_normalize_cpu(data, do_transpose=do_transpose)


def _subtract_mean_and_normalize_cpu(data, do_transpose=False):
    """
    Prep an array of cell x gene data for correlation distance
    computation.

    Parameters
    ----------
    data:
        A (n_cells, n_genes) np.ndarray
    do_transpose:
        A boolean. If False, return a (n_cells, n_gene)
        array. If True, return a (n_gene, n_cells) array

    Return
    ------
    data with the mean of each row subtracted and the
    each mean-subtracted row normalized to its L2 norm.
    (The array will also be transposed relative to the
    input if do_transpose)
    """
    mu = np.mean(data, axis=1)
    data = (data.transpose()-mu)
    norm = np.sqrt(np.sum(data**2, axis=0))

    # if norm=0, it means that whole cell had the same
    # value in all genes. Probably not interesting.
    # Set those norms to 1 to avoid divide by zero
    # problems
    invalid = (norm == 0.0)
    norm[invalid] = 1.0

    data = data/norm
    if not do_transpose:
        return data.transpose()
    return data


def _subtract_mean_and_normalize_gpu(data,
                                     do_transpose=False,
                                     gpu_index=0,
                                     timers=None):

    with torch.no_grad():

        t = time.time()
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).type(torch.float)
            data = data.to(device=f'cuda:{gpu_index}', non_blocking=True)
        update_timer("togpu", t, timers)

        t = time.time()
        mu = torch.mean(data, axis=1)
        data = torch.t(data)-mu
        del mu
        norm = torch.sqrt(torch.sum(data**2, axis=0))
        update_timer("meansqrt", t, timers)
        # if norm=0, it means that whole cell had the same
        # value in all genes. Probably not interesting.
        # Set those norms to 1 to avoid divide by zero
        # problems
        invalid = (norm == 0.0)
        norm[invalid] = 1.0

        data = data/norm
        del norm
    if not do_transpose:
        return torch.t(data)
    return data
