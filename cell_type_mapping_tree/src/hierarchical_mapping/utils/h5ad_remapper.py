import h5py
import multiprocessing
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

    t_write = 0.0
    with h5py.File(h5ad_path, 'r', swmr=True) as h5ad_handle:
        h5ad_server = H5adServer(
            h5ad_handle=h5ad_handle,
            buffer_size=read_in_size)

        while True:
            h5ad_data = h5ad_server.update()
            if h5ad_data is None:
                break


            t0 = time.time()
            process_list = []
            for collector_obj in row_collector_list:
                p = multiprocessing.Process(
                        target=_hunter_gather_worker,
                        kwargs={
                            'data_obj': h5ad_data,
                            'collector_obj': collector_obj})
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()
            duration = time.time()-t0
            t_write += duration
            print(f"row {h5ad_server.r0} -- "
                  f"spent {h5ad_server.t_load/3600.0:.2e} hrs reading; "
                  f"{t_write/3600.0:.2e} hrs writing")

    with zarr.open(output_path, 'a') as zarr_handle:
        zarr_handle['indptr'][:] = new_indptr

    duration = (time.time()-t0)/3600.0
    print(f"whole process took {duration:.2e} hrs")


def _hunter_gather_worker(
        data_obj,
        collector_obj):
    collector_obj.ingest_data(h5ad_server=data_obj)


class DataObject(object):

    def __init__(
            self,
            base_row,
            data,
            indices,
            indptr):

        self.base_row=base_row
        self.data=data
        self.indices=indices
        self.indptr=indptr


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
            return None
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
        self._base_row = self.r0
        i0 = self._raw_indptr[self.r0]
        i1 = self._raw_indptr[r1]

        self._data[:i1-i0] = self.h5ad_handle['X']['data'][i0:i1]
        self._indices[:i1-i0] = self.h5ad_handle['X']['indices'][i0:i1]
        self._valid_idx = i1-i0

        self._indptr_chunk = self._raw_indptr[self.r0:r1+1]-self._raw_indptr[self.r0]
        self.r0 = r1
        self.t_load += time.time()-t0

        return DataObject(
                    base_row=self.base_row,
                    data=self.data,
                    indices=self.indices,
                    indptr=self.indptr)


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
        self._new_row_to_idx = new_indptr

        self._row_chunk = row_chunk
        self._already_written = set()
        self._buffer_size = buffer_size
        self._data_dtype = data_dtype

    @property
    def row_ct(self):
        return self._ct_rows

    @property
    def is_complete(self):
        return self._complete

    def ingest_data(
            self,
            h5ad_server):

        self._data = np.zeros(self._buffer_size, dtype=self._data_dtype)
        self._indices = np.zeros(self._buffer_size, dtype=int)
        self._output_idx = np.zeros(self._buffer_size, dtype=int)
        self._stored = 0

        data_chunk = h5ad_server.data
        indices_chunk = h5ad_server.indices
        indptr_chunk = h5ad_server.indptr
        r0 = h5ad_server.base_row

        t0 = time.time()
        i0 = indptr_chunk[0]
        for r_idx in range(len(indptr_chunk)-1):
            old_row = r_idx + r0
            new_row = self._old_row_to_new_row[old_row]
            if new_row < self._row_chunk[0] or new_row >= self._row_chunk[1]:
                continue
            if new_row in self._already_written:
                continue

            data_i0 = indptr_chunk[r_idx] + i0
            data_i1 = indptr_chunk[r_idx+1] + i0
            delta = data_i1-data_i0
            if self._stored + delta > self._buffer_size:
                self._flush()

            self._data[self._stored:
                       self._stored+delta] = data_chunk[data_i0:data_i1]
            self._indices[self._stored:
                          self._stored+delta] = indices_chunk[data_i0:data_i1]

            self._output_idx[
                self._stored:
                self._stored+delta] = np.arange(self._new_row_to_idx[new_row],
                                                self._new_row_to_idx[new_row+1]).astype(int)

            self._stored += delta
            self._ct_rows += 1
            self._already_written.add(new_row)

        self._flush()

        if len(self._already_written) == (self._row_chunk[1]-self._row_chunk[0]):
            if self._stored > 0:
                self._flush()
            self._complete = True

        self._data = None
        self._indices = None
        self._ouptut_idx = None

        self.t_write += time.time()-t0

    def _flush(self):
        """
        Write buffers to output
        """
        with zarr.open(self._zarr_path, 'a') as out_handle:
            out_handle['data'][self._output_idx[:self._stored]] = self._data[:self._stored]
            out_handle['indices'][self._output_idx[:self._stored]] = self._indices[:self._stored]

        self._stored = 0
        print_timing(t0=self._t0,
                     i_chunk=self._ct_rows,
                     tot_chunks=self._tot_rows,
                     unit='hr')
