import numpy as np


def correlation_nearest_neighbors(
        baseline_array,
        query_array):
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

    Returns
    -------
    A (n_cells_1, ) np.ndarray of integers representing
    the index of the nearest cell from baseline_arry to each
    cell in query_array (i.e. the returned value at 11 is the
    nearest neighbor of query_array[11, :])
    """
    return np.argmax(correlation_dot(baseline_array, query_array), axis=0)


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

def correlation_dot(arr0, arr1):
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
    """
    arr0 = _subtract_mean_and_normalize(arr0, do_transpose=False)
    arr1 = _subtract_mean_and_normalize(arr1, do_transpose=True)
    return np.dot(arr0, arr1)


def _subtract_mean_and_normalize(data, do_transpose=False):
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
    data = data/norm
    if not do_transpose:
        return data.transpose()
    return data
