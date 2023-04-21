import h5py
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

    This has been optimized for n_cols >> n_rows
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

    def __eq__(self, other):
        if self.n_rows != other.n_rows:
            return False
        if self.n_cols != other.n_cols:
            return False
        if not np.array_equal(self.data[:, :self.n_ints-1],
                              other.data[:, :self.n_ints-1]):
            return False

        for i_col in range(8*(self.n_cols//8), self.n_cols, 1):
            if not np.array_equal(self.get_col(i_col),
                                  other.get_col(i_col)):
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def set_col(self, i_col, data):
        """
        Set the specified column to the booleans in data
        """
        (i_int,
         val) = self._col_to_int_val(i_col)
        current_column = (self.data[:, i_int] & val).astype(bool)

        flip_on = (~current_column) & data
        self.data[:, i_int][flip_on] += val

        flip_off = current_column & (~data)
        self.data[:, i_int][flip_off] -= val

    def get_row(self, i_row):
        """
        Return a row as a boolean array
        """
        return unpack_binarized_boolean_array(
            binarized_data=self.data[i_row, :],
            n_booleans=self.n_cols)

    def get_row_batch(self, row0, row1):
        """
        Return an array of boolean rows
        """
        result = np.zeros((row1-row0, self.n_cols), dtype=bool)
        for i_row in range(row0, row1, 1):
            result[i_row-row0, :] = self.get_row(i_row)
        return result

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

    def row_sum(self):
        """
        Return a 1-D numpy array (length=self.n_cols) representing
        the sum of the booleans across rows
        """
        result = np.zeros(self.n_cols, dtype=int)
        for i_row in range(self.n_rows):
            row = self.get_row(i_row)
            result += row
        return result

    def col_sum(self):
        """
        Return a 1-D numpy array (length=self.n_rows) representing
        the sum of the booleans across rows
        """
        result = np.zeros(self.n_rows, dtype=int)
        for i_row in range(self.n_rows):
            result[i_row] = self.get_row(i_row).sum()
        return result

    def _verify_other_same_shape(
            self,
            other):
        """
        Raise an error if other (another BinarizedBooleanArray) is not
        the same shape as this one
        """
        if self.n_rows != other.n_rows:
            raise RuntimeError(
                "n_rows do not match between BinaraizedBooleanArrays\n"
                f"{self.n_rows} != {other.n_rows}")

        if self.n_cols != other.n_cols:
            raise RuntimeError(
                "n_cols do not match between BinaraizedBooleanArrays\n"
                f"{self.n_cols} != {other.n_cols}")

        if self.n_ints != other.n_ints:
            raise RuntimeError(
                "n_ints do not match between BinaraizedBooleanArrays\n"
                f"{self.n_ints} != {other.n_ints}")

    def add_other(self, other):
        """
        Add the data matrix from another BinarizedBooleanArray
        to this matrix.

        Really this is just doing a bitwise or row by row of the
        two matrices
        """
        self._verify_other_same_shape(other)

        for i_row in range(self.n_rows):
            this_row = self.get_row(i_row)
            other_row = other.get_row(i_row)
            self.set_row(i_row, this_row | other_row)

    def copy_columns_from_other(self, other, col_span):
        """
        Copy columns col_span[0]:col_span[1] from other
        (another BinarizedBooleanArray) tho this BinarizedBooleanArray
        """
        self._verify_other_same_shape(other)

        # first just copy the whole ints that can be copied over
        n_int0 = np.ceil(col_span[0]/8).astype(int)
        n_int1 = np.floor(col_span[1]/8).astype(int)
        if n_int1 > n_int0:
            self.data[:, n_int0:n_int1] = other.data[:, n_int0:n_int1]

        remainder_spans = []
        n_int_col0 = n_int0*8
        if col_span[0] < n_int_col0:
            remainder_spans.append((col_span[0], n_int_col0))
        n_int_col1 = n_int1*8
        if col_span[1] > n_int_col1:
            remainder_spans.append((n_int_col1, col_span[1]))

        for span in remainder_spans:
            for i_col in range(span[0], span[1], 1):
                other_col = other.get_col(i_col)
                self.set_col(i_col, other_col)

    def copy_other_as_columns(self, other, col0):
        """
        Set columns [col0:col0+other.n_cols] from other.
        Note: col0 *must* be an integer multiple of 8.
        """
        if col0 % 8 != 0:
            raise RuntimeError(
                "col0 must be integer multiple of 8\n"
                f"{col0} % 8 = {col0%8}")

        if col0+other.n_cols > self.n_cols:
            raise RuntimeError(
                f"col0: {col0}\nother.n_cols {other.n_cols}\n"
                f"but self only has {self.n_cols} columns")

        if other.n_rows != self.n_rows:
            raise RuntimeError(
                "self.n_rows != other.n_rows\n"
                f"{self.n_rows} != {other.n_rows}")

        this_int0 = col0//8
        other_int1 = np.floor(other.n_cols/8).astype(int)
        self.data[:,
                  this_int0:this_int0+other_int1] = other.data[:,
                                                               :other_int1]
        if other.n_cols % 8 != 0:
            this_col0 = this_int0 * 8
            other_col0 = other_int1 * 8
            for i_col in range(other_col0, other.n_cols, 1):
                col = other.get_col(i_col)
                self.set_col(this_col0+i_col, col)

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

    def write_to_h5(self, h5_path, h5_group):
        """
        Record this BinarizedBooleanArray to an HDF5 file.

        Parameters
        ----------
        h5_path:
           Path to the file that will be written to (will be
           opened in 'append' mode so that many arrays can
           be stored in the same file, as long as they have
           different group names).
       h5_group:
           The group under which this array will be stored
        """
        with h5py.File(h5_path, 'a') as out_file:
            out_file.create_dataset(
                f'{h5_group}/n_cols', data=self.n_cols)
            out_file.create_dataset(
                f'{h5_group}/data',
                data=self.data,
                chunks=(min(self.n_rows, 1000),
                        min(self.n_ints, 1000)),
                compression='gzip')

    @classmethod
    def read_from_h5(cls, h5_path, h5_group):
        """
        Reconstitute the BinarizedBooleanArray from an HDF5 file

        Parameters
        ----------
        h5_path:
            Path to the file to be read.
        h5_group:
            The group in the HDF5 file where the data is kept
        """
        with h5py.File(h5_path, 'r') as in_file:
            n_cols = in_file[f'{h5_group}/n_cols'][()]
            data = in_file[f'{h5_group}/data'][()]
        return cls.from_data_array(
            data_array=data,
            n_cols=n_cols)
