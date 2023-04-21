import numpy as np


def binarize_boolean_array(
        data):
    """
    Convert a 1-D numpy array of booleans into a bit-packed
    numpy array of np.uint8
    """
    elements = len(data)
    out_elements = np.ceil(elements/8).astype(int)
    result = np.zeros(out_elements, dtype=np.uint8)
    pwr = np.uint8(0)
    factor = np.uint8(2)

    new_rows = np.ceil(elements/8).astype(int)
    total_el = 8*new_rows
    if total_el > len(data):
        new_data = np.zeros(total_el, dtype=bool)
        new_data[:len(data)] = data
        data = new_data
        del new_data

    data = data.reshape((new_rows, 8))
    result = np.zeros(out_elements, dtype=np.uint8)

    for i_col in range(8):
        if pwr == 0:
            pwr = np.uint8(1)
        else:
            pwr *= factor
        this_col = data[:, i_col]
        result[this_col] += pwr
    return result


def unpack_binarized_boolean_array(
        binarized_data,
        n_booleans):
    """
    Convert a bit-packed array of np.uint8 back into an
    array of booleans.

    Parameters
    ----------
    binarized_data:
        1-D np.array of np.uint8 to be unpacked
    n_booleans:
        expected size of output array (since the array of
        np.uint8 could have trailing bits)

    Returns
    -------
    1-D numpy array of booleans
    """
    n_rows = len(binarized_data)
    result = np.zeros((n_rows, 8), dtype=bool)
    pwr = np.uint8(1)
    factor = np.uint8(2)
    for ii in range(8):
        valid = (binarized_data & pwr > 0)
        result[:, ii][valid] = True
        if ii < 7:
            pwr *= factor
    return result.flatten()[:n_booleans]


def unpack_binarized_boolean_array_2D(
        binarized_data,
        n_booleans):
    """
    Convert a bit-packed array of np.uint8 back into an
    array of booleans.

    Parameters
    ----------
    binarized_data:
        2-D np.array of np.uint8 to be unpacked
    n_booleans:
        number of booleans per row of the output
        (since the array of np.uint8 could have trailing bits)

    Returns
    -------
    2-D numpy array of booleans
    """
    n_int = binarized_data.shape[1]
    result = np.zeros((binarized_data.shape[0], n_int, 8), dtype=bool)
    pwr = np.uint8(1)
    factor = np.uint8(2)

    for ii in range(8):
        valid = (binarized_data & pwr > 0)
        result[:, :, ii][valid] = True
        if ii < 7:
            pwr *= factor
    result = result.reshape(binarized_data.shape[0], n_int*8)
    result = result[:, :n_booleans]
    return result
