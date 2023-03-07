import zarr
import numpy as np
import pathlib


class SparseZarrWriter(object):

    def __init__(self, file_path):
        self._file_path = pathlib.Path(file_path)

    def write_data(
            self,
            i0: int,
            i1: int,
            data_chunk: np.ndarray,
            indices_chunk: np.ndarray):
        with zarr.open(self._file_path, 'a') as out_handle:
            out_handle['data'][i0:i1] = data_chunk
            out_handle['indices'][i0:i1] = indices_chunk

    def write_indptr(self, new_indptr):
        with zarr.open(self._file_path, 'a') as out_handle:
            out_handle['indptr'][:] = new_indptr


class ArrayWriter(object):

    def __init__(self, data, indices, indptr):
        self._data = data
        self._indices = indices
        self._indptr = indptr

    def write_data(
            self,
            i0: int,
            i1: int,
            data_chunk: np.ndarray,
            indices_chunk: np.ndarray):

        self._data[i0:i1] = data_chunk
        self._indices[i0:i1] = indices_chunk

    def write_indptr(self, new_indptr):
        self._indptr[:] = new_indptr
