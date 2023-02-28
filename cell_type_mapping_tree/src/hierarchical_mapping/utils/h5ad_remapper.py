import h5py
import numpy as np
import pathlib
import time
import zarr

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.sparse_utils import (
    precompute_indptr)

from hierarchical_mapping.utils.sparse_zarr_utils import (
    _create_empty_zarr)


def rearrange_sparse_h5ad_hunter_gather(
        h5ad_path,
        output_path,
        row_order,
        chunks=5000,
        n_row_collectors=5,
        buffer_size=10000000,
        read_in_size=10000000,
        verbose=True):

    t0 = time.time()
    with h5py.File(h5ad_path, 'r', swmr=True) as input_handle:
        old_indptr = input_handle['X']['indptr'][()]
        new_indptr = precompute_indptr(
                        indptr_in=old_indptr,
                        row_order=row_order)

        data_shape = input_handle['X']['data'].shape
        data_dtype = input_handle['X']['data'].dtype

        _create_empty_zarr(
             data_shape=data_shape,
             indptr_shape=old_indptr.shape,
             output_path=output_path,
             data_dtype=data_dtype,
             chunks=chunks)

    row_collector_list = []
    n_rows = len(row_order)
    r_per_collector = np.round(n_rows/n_row_collectors).astype(int)
    for ii in range(n_row_collectors):
        r0 = ii*r_per_collector
        if ii == n_row_collectors-1:
            r1 = n_rows
        else:
            r1 = r0+r_per_collector

        collector = RowCollector(
                        zarr_path=output_path,
                        new_row_order=row_order,
                        new_indptr=new_indptr,
                        row_chunk=(r0, r1),
                        buffer_size=buffer_size)

        row_collector_list.append(collector)

    with h5py.File(h5ad_path, 'r', swmr=True) as h5ad_handle:
        h5ad_server = H5adServer(
            h5ad_handle=h5ad_handle,
            buffer_size=read_in_size)
        keep_going = True
        while keep_going:
            data = h5ad_server.update()
            for collector in row_collector_list:
               if collector.is_complete:
                    continue
               collector.ingest_data(
                    h5ad_server=h5ad_server)

            keep_going = False
            t_write = 0.0
            for collector in row_collector_list:
                t_write += collector.t_write
                if not collector.is_complete:
                    keep_going = True

            duration = (time.time()-t0)/3600.0
            if verbose:
                print(f"spent {duration:.2e} hrs total; "
                      f"{h5ad_server.t_load/3600.0:.2e} hrs reading; "
                      f"{t_write/3600.0:.2e} hrs writing -- "
                      f"reading row {h5ad_server.r0:.2e}")


    with zarr.open(output_path, 'a') as zarr_handle:
        zarr_handle['indptr'][:] = new_indptr

    duration = (time.time()-t0)/3600.0
    print(f"whole process took {duration:.2e} hrs")


class H5adServer(object):

    def __init__(
            self,
            h5ad_handle,
            buffer_size):
        """
        read_in_size is the number of data elements to serve up at once
        """
        self.h5ad_handle = h5ad_handle
        self._raw_indptr = h5ad_handle['X']['indptr'][:]
        self._data = np.zeros(
                        buffer_size,
                        dtype=h5ad_handle['X']['data'].dtype)
        self._indices = np.zeros(buffer_size, dtype=int)
        self.r0 = 0
        self.buffer_size = buffer_size
        self.t_load = 0.0

    @property
    def data(self):
        return self._data[:self._valid_idx]

    @property
    def indices(self):
        return self._indices[:self._valid_idx]

    @property
    def indptr(self):
        return self._indptr_chunk

    @property
    def base_row(self):
        return self._base_row

    def update(self):
        t0 = time.time()
        if self.r0 == len(self._raw_indptr)-1:
            self.r0 = 0
        projected_buffer = 0
        r1 = self.r0
        for candidate in range(self.r0+1, len(self._raw_indptr), 1):
            delta = self._raw_indptr[candidate] - self._raw_indptr[r1]
            if projected_buffer + delta > self.buffer_size:
                break
            projected_buffer += delta
            r1 = candidate
        if r1 == self.r0:
            raise RuntimeError(
                "could not load h5ad data with buffer "
                f"{self.buffer_size}")
        result = dict()
        self._base_row = self.r0
        i0 = self._raw_indptr[self.r0]
        i1 = self._raw_indptr[r1]

        self._data[:i1-i0] = self.h5ad_handle['X']['data'][i0:i1]
        self._indices[:i1-i0] = self.h5ad_handle['X']['indices'][i0:i1]
        self._valid_idx = i1-i0

        self._indptr_chunk = self._raw_indptr[self.r0:r1+1]-self._raw_indptr[self.r0]
        self.r0 = r1
        self.t_load += time.time()-t0
        return result


class RowCollector(object):

    def __init__(
           self,
           zarr_path,
           new_row_order,
           new_indptr,
           row_chunk,
           buffer_size,
           data_dtype=np.float32):
        """
        row_chunk is (row_min, row_max) that this
        collector is looking for (in new_row coordinates)

        buffer_size is maximum number of data/indices
        elements to be stored at a time (must be greater
        than number of columns in array)
        """
        self.t_write = 0.0
        self._t0 = time.time()
        self._tot_rows = row_chunk[1]-row_chunk[0]
        self._ct_rows = 0

        self._complete = False
        self._zarr_path = pathlib.Path(zarr_path)
        if not self._zarr_path.is_dir():
            raise RuntimeError(
                f"{self._zarr_path} is not dir")

        self._old_row_to_new_row = dict()
        self._new_row_to_old_row = new_row_order
        for ii, rr in enumerate(new_row_order):
            self._old_row_to_new_row[rr] = ii

        self._row_chunk = row_chunk
        self._buffer_size = buffer_size
        self._data = np.zeros(buffer_size, dtype=data_dtype)
        self._indices = np.zeros(buffer_size, dtype=int)
        self._buffer_mask = np.zeros(buffer_size, dtype=bool)

        self._new_row_to_idx = new_indptr
        self._current_chunk = None
        self._set_next_chunk()

    @property
    def is_complete(self):
        return self._complete

    def ingest_data(
            self,
            h5ad_server):

        data_chunk = h5ad_server.data
        indices_chunk = h5ad_server.indices
        indptr_chunk = h5ad_server.indptr
        r0 = h5ad_server.base_row

        t0 = time.time()
        i0 = indptr_chunk[0]
        for r_idx in range(len(indptr_chunk)-1):
            old_row = r_idx + r0
            new_row = self._old_row_to_new_row[old_row]
            if new_row < self._current_chunk[0] or new_row >= self._current_chunk[1]:
                continue
            buffer_idx = self._old_row_to_buffer[old_row]
            data_i0 = indptr_chunk[r_idx] + i0
            data_i1 = indptr_chunk[r_idx+1] + i0
            delta = data_i1-data_i0
            self._data[buffer_idx:
                       buffer_idx+delta] = data_chunk[data_i0:data_i1]
            self._indices[buffer_idx:
                          buffer_idx+delta] = indices_chunk[data_i0:data_i1]
            self._buffer_mask[buffer_idx:buffer_idx+delta] = True

        if self._buffer_mask.sum() == self._current_buffer_size:
            self._flush()
            self._set_next_chunk()

        self.t_write += time.time()-t0

    def _flush(self):
        """
        Write buffers to output
        """
        z0 = self._new_row_to_idx[self._current_chunk[0]]
        z1 = self._new_row_to_idx[self._current_chunk[1]]
        with zarr.open(self._zarr_path, 'a') as out_handle:
            out_handle['data'][z0:z1] = self._data[:self._current_buffer_size]
            out_handle['indices'][z0:z1] = self._indices[
                                             :self._current_buffer_size]

        self._ct_rows += self._current_chunk[1]-self._current_chunk[0]
        print_timing(t0=self._t0,
                     i_chunk=self._ct_rows,
                     tot_chunks=self._tot_rows,
                     unit='hr')

    def _set_next_chunk(self):
        """
        set self._current_chunk to (row_min, row_max)
        such that the buffers will accommodate the
        required data
        """
        self._current_buffer_size = None
        self._old_row_to_buffer = None
        self._buffer_mask[:] = False

        if self._current_chunk is None:
            r0 = self._row_chunk[0]
        else:
            r0 = self._current_chunk[1]

        if r0 == self._row_chunk[1]:
            self._complete = True
            self._current_chunk = (self._row_chunk[1], self._row_chunk[1])
            return

        r1 = r0
        projected_buffer = 0
        for candidate in range(r0+1, self._row_chunk[1]+1, 1):
            delta = self._new_row_to_idx[candidate] - self._new_row_to_idx[r1]
            if projected_buffer + delta > self._buffer_size:
                break
            r1 = candidate
            projected_buffer += delta

        if r1 == r0:
            raise RuntimeError(
                f"at r0={r0}, could not assign new buffer for "
                f"global chunk {self._row_chunk}")

        # how big is the buffer we need for current chunk
        self._current_buffer_size = projected_buffer

        # chunk we are focusing on ingesting now
        self._current_chunk = (r0, r1)

        # map from
        self._old_row_to_buffer = dict()
        buffer0 = self._new_row_to_idx[r0]
        for new_row in range(r0, r1):
            old_row = self._new_row_to_old_row[new_row]
            self._old_row_to_buffer[old_row] = self._new_row_to_idx[new_row]-buffer0
        n = self._current_chunk[1]-self._current_chunk[0]
        print(f"looking to write {n} rows")
