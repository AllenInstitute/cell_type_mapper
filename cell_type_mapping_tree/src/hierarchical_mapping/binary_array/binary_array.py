import numpy as np

from hierarchical_mapping.binary_array.utils import (
    binarize_boolean_array,
    unpack_binarized_boolean_array)


class BinarizedBooleanArray(object):
    """
    A class for bit packing an array of booleans into an array of np.uint8.

    Parameters
    ----------
    n_rows
        Number of rows in the boolean array to be binarized

    n_cols
        Number of columns in the boolean array to be binarized

    Notes
    -----
    The constructor will just initialize an array of np.uints of the
    desired shape that are all zeros.
    """
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

        # number of np.uint columns to be stored
        self.n_ints = np.ceil(n_cols/8).astype(int)
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
        
