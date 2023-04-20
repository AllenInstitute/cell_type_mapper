import h5py
import numpy as np
import pathlib

from hierarchical_mapping.binary_array.binary_array import (
    n_int_from_n_cols,
    BinarizedBooleanArray)


class BackedBinarizedBooleanArray(object):
    """
    Version of BinarizedBooleanArray that leaves its data
    on disk

    Parameters
    ----------
    h5_path:
        Path to the HDF5 file where backed data will be stored.
        (Raises exception if already exists)

    n_rows
        Number of rows in the boolean array to be binarized

    n_cols
        Number of columns in the boolean array to be binarized

    Notes
    -----
    Right now, can only be populated using copy_other_as_columns
    """

    def __init__(
            self,
            h5_path,
            n_rows,
            n_cols):
        self.h5_path = pathlib.Path(h5_path)

        if self.h5_path.exists():
            raise RuntimeError(
                f"{self.h5_path} already exists")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_ints = n_int_from_n_cols(n_cols)
        with h5py.File(self.h5_path, 'w') as out_file:
            out_file.create_dataset(
                'data',
                data=np.zeros((self.n_rows, self.n_ints), dtype=np.uint8),
                chunks=(min(self.n_rows, 1000), min(self.n_ints, 1000)))

        self.loaded_chunk = None
        self.chunk_has_changed = False

        # how many rows to load at a time
        self._load_row_size = 8000000//n_cols

        # how many columns to load at a time
        self._load_col_size = 8000000//n_rows

        # mapping from the bit in a given np.uint8 to its value
        self.bit_lookup = {
            ii: np.uint8(2**ii) for ii in range(8)}

    def __eq__(self, other):
        raise NotImplementedError(
            "do not have __eq__ for BackedBinarizedBooleanArray")

    def __ne__(self, other):
        raise NotImplementedError(
            "do not have __ne__ for BackedBinarizedBooleanArray")

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

    def _int_to_col(self, i_int):
        return min(self.n_cols, i_int*8)

    def _load_chunk(
            self,
            row_spec,
            col_spec):
        """
        Load a chunk of data into memory.

        Parameters
        ----------
        row_spec:
            Tuple of form (row_min, row_max)
        col_spec:
            Tuple of form (col_min, col_max)
        """
        if self.loaded_chunk is not None:
            if self.chunk_has_changed:
                self._write_chunk_back()

        int_min = self._col_to_int(col_spec[0])
        int_max = min(self.n_ints, self._col_to_int(col_spec[1])+1)
        with h5py.File(self.h5_path, 'r') as in_file:
            chunk = in_file['data'][
                    row_spec[0]:row_spec[1],
                    int_min:int_max]

        actual_cols = (self._int_to_col(int_min),
                       self._int_to_col(int_max))
        chunk = BinarizedBooleanArray.from_data_array(
                    data_array=chunk,
                    n_cols=actual_cols[1]-actual_cols[0])

        self.loaded_chunk = {
            "rows": row_spec,
            "cols": actual_cols,
            "ints": (int_min, int_max),
            "data": chunk}
        self.chunk_has_changed = False

    def _write_chunk_back(self):
        """
        Write the current loaded chunk back to disk
        """
        if self.loaded_chunk is None:
            return
        rows = self.loaded_chunk['rows']
        cols = self.loaded_chunk['ints']
        data = self.loaded_chunk['data'].data

        with h5py.File(self.h5_path, "a") as out_file:
            out_file['data'][
                rows[0]:rows[1],
                cols[0]:cols[1]] = data
        self.chunk_has_changed = False

    def _need_to_load_col(self, i_col):
        """
        Do I need to load a chunkt to get the current column?
        """
        if self.loaded_chunk is None:
            return True

        if self.loaded_chunk['rows'][0] != 0:
            return True
        elif self.loaded_chunk['rows'][1] != self.n_rows:
            return True

        loaded_cols = self.loaded_chunk['cols']
        if i_col >= loaded_cols[0] and i_col < loaded_cols[1]:
            return False
        return True

    def _need_to_load_row(self, i_row):
        """
        Do I need to load a chunkt to get the current row?
        """
        if self.loaded_chunk is None:
            return True

        if self.loaded_chunk['cols'][0] != 0:
            return True
        elif self.loaded_chunk['cols'][1] != self.n_cols:
            return True

        loaded_rows = self.loaded_chunk['rows']
        if i_row >= loaded_rows[0] and i_row < loaded_rows[1]:
            return False
        return True

    def _change_whole_col(self, i_col, set_to):
        """
        set_to is either True or False
        i_col is the column that needs to be set
        """
        need_to_load = self._need_to_load_col(i_col)
        if need_to_load:
            self._load_chunk(
                row_spec=(0, self.n_rows),
                col_spec=(i_col, min(self.n_cols,
                                     i_col+self._load_col_size)))

        mapped_col = i_col - self.loaded_chunk['cols'][0]
        if set_to:
            self.loaded_chunk['data'].set_col_true(mapped_col)
        else:
            self.loaded_chunk['data'].set_col_false(mapped_col)
        self.chunk_has_changed = True

    def set_col_true(self, i_col):
        self._change_whole_col(i_col, set_to=True)

    def set_col_false(self, i_col):
        self._change_whole_col(i_col, set_to=False)

    def _change_whole_row(self, i_row, set_to):
        """
        set_to is either True or False
        i_row is the row that needs to be set
        """
        need_to_load = self._need_to_load_row(i_row)
        if need_to_load:
            self._load_chunk(
                row_spec=(i_row, min(self.n_rows,
                                     i_row+self._load_row_size)),
                col_spec=(0, self.n_cols))

        mapped_row = i_row - self.loaded_chunk['rows'][0]
        if set_to:
            self.loaded_chunk['data'].set_row_true(mapped_row)
        else:
            self.loaded_chunk['data'].set_row_false(mapped_row)
        self.chunk_has_changed = True

    def set_row_true(self, i_row):
        self._change_whole_row(i_row, set_to=True)

    def set_row_false(self, i_row):
        self._change_whole_row(i_row, set_to=False)

    def get_col(self, i_col):
        need_to_load = self._need_to_load_col(i_col)
        if need_to_load:
            self._load_chunk(
                row_spec=(0, self.n_rows),
                col_spec=(i_col, min(self.n_cols, i_col+1000)))
        mapped_col = i_col-self.loaded_chunk['cols'][0]
        return self.loaded_chunk['data'].get_col(mapped_col)

    def get_row(self, i_row):
        need_to_load = self._need_to_load_row(i_row)
        if need_to_load:
            self._load_chunk(
                row_spec=(i_row, min(self.n_rows, i_row+1000)),
                col_spec=(0, self.n_cols))
        mapped_row = i_row-self.loaded_chunk['rows'][0]
        return self.loaded_chunk['data'].get_row(mapped_row)
