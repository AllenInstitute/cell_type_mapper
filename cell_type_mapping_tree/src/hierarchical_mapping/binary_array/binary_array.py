import numpy as np

from hierarchical_mapping.binary_array.utils import (
    binarize_boolean_array,
    unpack_binarized_boolean_array)


def n_int_from_n_cols(n_cols):
    """
    How many np.uint8 are needed to pack in n_cols booleans
    """
    return np.ceil(n_cols/8).astype(int)


class BinarizedBooleanArray(object):
    """
    A class for bit packing an array of booleans into an array of np.uint8.

    Parameters
    ----------
    n_rows
        Number of rows in the boolean array to be binarized

    n_cols
        Number of columns in the boolean array to be binarized
    initialize_data:
        A boolean. If True, create an array of zeros backing this
        object up. This should only be false when instantiating
        this class with the from_data_array method.

    Notes
    -----
    The constructor will just initialize an array of np.uints of the
    desired shape that are all zeros.
    """
    def __init__(self, n_rows, n_cols, initialize_data=True):
        self.n_rows = n_rows
        self.n_cols = n_cols

        # number of np.uint columns to be stored
        self.n_ints = n_int_from_n_cols(n_cols)
        if initialize_data:
            self.data = np.zeros((self.n_rows, self.n_ints), dtype=np.uint8)

        # mapping from the bit in a given np.uint8 to its value
        self.bit_lookup = {
            ii: np.uint8(2**ii) for ii in range(8)}

    def _col_to_int(self, i_col):
        """
        Convert i_col, the index of the column in the full
        boolean array, into i_int, the index of the column
        in self.data
        """
        return i_col // 8

    def _col_to_bit(self, i_col):
        """
        Convert i_col, the index of the column in the full
        boolean array, into i_bit, the index of the bit
        within the corresponding np.uint8.
        """
        return i_col % 8

    def _col_to_int_val(self, i_col):
        """
        Convert i_col, the index of a column in the full boolean array,
        into i_int, the index of the column in self.data and val,
        the value of the bit i_col respresented
        """
        i_int = self._col_to_int(i_col)
        i_bit = self._col_to_bit(i_col)
        val = self.bit_lookup[i_bit]
        return (i_int, val)

    def set_row(self, i_row, data):
        """
        Set the values in the specified row.

        Parameters
        ----------
        i_row:
            index of the row
        data:
            np.array of booleans
        """
        if data.shape != (self.n_cols, ):
            raise ValueError(
                f"self.n_cols is {self.n_cols}\n"
                f"you passed in a row of shape {data.shape}")
        self.data[i_row, :] = binarize_boolean_array(data)

    def get_row(self, i_row):
        """
        Return a row as a boolean array
        """
        return unpack_binarized_boolean_array(
            binarized_data=self.data[i_row, :],
            n_booleans=self.n_cols)

    def get_col(self, i_col):
        """
        Return a column as a boolean array
        """
        (i_int,
         val) = self._col_to_int_val(i_col)
        return (self.data[:, i_int] & val).astype(bool)

    def set_col_false(self, i_col):
        """
        Set the column specified by i_col to False in all rows
        """
        (i_int,
         val) = self._col_to_int_val(i_col)
        operand = 255-val
        self.data[:, i_int] &= operand

    def set_col_true(self, i_col):
        """
        Set the column specified by i_col to True in all rows
        """
        (i_int,
         val) = self._col_to_int_val(i_col)
        valid = (self.data[:, i_int] & val == 0)
        self.data[:, i_int][valid] += val

    def set_row_false(self, i_row):
        """
        Set the row specified by i_row to False in all columns
        """
        self.data[i_row, :] = 0

    def set_row_true(self, i_row):
        """
        Set the row specified by i_row to True in all columns
        """
        self.data[i_row, :] = np.iinfo(np.uint8).max

    @classmethod
    def from_data_array(
            cls,
            data_array,
            n_cols):
        """
        Constitute a BinarizedBooleanArray from an array of
        np.uint8.

        Parameters
        ----------
        data_array:
            array of np.uint8 that will back the
            resulting BinarizedBooleanArray
        n_cols:
            Number of cols in the unpacked boolean array (number of rows is
            inferred from shape of data_array)
        """
        if data_array.dtype != np.uint8:
            raise ValueError(
                "data_array must be of type np.uint8\n"
                f"you gave {data_array.dtype}")
        n_rows = data_array.shape[0]
        n_int = n_int_from_n_cols(n_cols)
        if data_array.shape[1] != n_int:
            raise ValueError(
                "You say you are loading a boolean array with"
                f"{n_cols} columns. That implies a np.uint8 array "
                f"with {n_int} columns. Your data array has shape "
                f"{data_array.shape}")
        result = cls(n_rows=n_rows, n_cols=n_cols, initialize_data=False)
        result.data = data_array
        return result
