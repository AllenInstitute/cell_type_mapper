import h5py
import multiprocessing
import numpy as np
import os
import pathlib
import time
import zarr
import tempfile

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_dict)

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
        verbose=True,
        tmp_dir=None):

    global_t0 = time.time()
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
                        buffer_size=buffer_size,
                        tmp_dir=tmp_dir)

        row_collector_list.append(collector)

    t_write = 0.0
    process_dict = dict()
    with h5py.File(h5ad_path, 'r', swmr=True) as h5ad_handle:
        n_rows_total = len(h5ad_handle['X']['indptr'][()])-1
        h5ad_server = H5adServer(
            h5ad_handle=h5ad_handle,
            buffer_size=read_in_size)

        write_t0 = time.time()
        keep_going = True

        h5ad_data = h5ad_server.update()

        collectors_for_this_chunk = set()
        while h5ad_data is not None:
            for i_coll, collector_obj in enumerate(row_collector_list):
                if i_coll in collectors_for_this_chunk:
                    continue
                if i_coll in process_dict:
                    continue

                p = multiprocessing.Process(
                        target=_hunter_gather_worker,
                        kwargs={
                            'data_obj': h5ad_data,
                            'collector_obj': collector_obj})
                p.start()
                process_dict[i_coll] = p
                collectors_for_this_chunk.add(i_coll)

            if len(collectors_for_this_chunk) == len(row_collector_list):
                print_timing(
                    t0=global_t0,
                    i_chunk=h5ad_server.r0,
                    tot_chunks=n_rows_total,
                    unit='hr')

                h5ad_data = h5ad_server.update()
                collectors_for_this_chunk = set()

            while len(process_dict) >= len(row_collector_list):
                process_dict = winnow_process_dict(process_dict)


        for k in process_dict.keys():
            process_dict[k].join()


    print("collecting temp files together")
    with zarr.open(output_path, 'a') as zarr_handle:
        zarr_handle['indptr'][:] = new_indptr
        for collector_obj in row_collector_list:
            _t0 = time.time()
            with h5py.File(collector_obj.tmp_h5_path, 'r') as in_file:
                span = in_file['idx_span'][()]
                d = 100000000
                for i0 in range(span[0], span[1], d):
                    i1 = min(span[1], i0+d)
                    zarr_handle['indices'][i0:i1] = in_file['indices'][i0-span[0]:i1-span[0]]
                    zarr_handle['data'][i0:i1] = in_file['data'][i0-span[0]:i1-span[0]]
            if collector_obj.tmp_h5_path.exists():
                collector_obj.tmp_h5_path.unlink()
            duration = (time.time()-_t0)/3600.0
            print(f"this collector took {duration:.2e} hrs -- total {len(row_collector_list)}")

    duration = (time.time()-global_t0)/3600.0
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
           data_dtype=np.float32,
           tmp_dir=None):
        """
        row_chunk is (row_min, row_max) that this
        collector is looking for (in new_row coordinates)

        buffer_size is maximum number of data/indices
        elements to be stored at a time (must be greater
        than number of columns in array)
        """

        self.tmp_h5_path = tempfile.mkstemp(
                             dir=tmp_dir,
                             suffix='.h5')
        os.close(self.tmp_h5_path[0])
        self.tmp_h5_path = pathlib.Path(self.tmp_h5_path[1])
        print(f"writing to {self.tmp_h5_path.resolve().absolute()} -- {tmp_dir}")
        self.t_write = 0.0
        self._t0 = time.time()
        self._tot_rows = row_chunk[1]-row_chunk[0]
        self._ct_rows = 0

        self._complete = False

        self._old_row_to_new_row = dict()
        self._new_row_to_old_row = new_row_order
        for ii, rr in enumerate(new_row_order):
            self._old_row_to_new_row[rr] = ii
        self._new_row_to_idx = new_indptr

        self._row_chunk = row_chunk

        idx_min = self._new_row_to_idx[self._row_chunk[0]]
        idx_max = self._new_row_to_idx[self._row_chunk[1]]
        n_el = idx_max-idx_min
        self.idx_min = idx_min

        with h5py.File(self.tmp_h5_path, 'w') as out_file:
            out_file.create_dataset(
                'data',
                shape=(n_el,),
                chunks=(min(n_el, 100000),),
                dtype=data_dtype,
                compression='gzip')
            out_file.create_dataset(
                'indices',
                shape=(n_el,),
                chunks=(min(n_el, 100000),),
                dtype=int,
                compression='gzip')

            out_file.create_dataset('idx_span',
                                    data=[idx_min, idx_max])

        self._already_written = set()
        self._buffer_size = buffer_size
        self._data_dtype = data_dtype

    def __del__(self):
        if self.tmp_h5_path.exists():
            self.tmp_h5_path.unlink()

    @property
    def row_ct(self):
        return self._ct_rows

    @property
    def is_complete(self):
        return self._complete

    def ingest_data(
            self,
            h5ad_server):
        with h5py.File(self.tmp_h5_path, 'a') as out_file:
            self._ingest_data(h5ad_server=h5ad_server, file_handle=out_file)

    def _ingest_data(
            self,
            h5ad_server,
            file_handle):
        t0 = time.time()
        data_chunk = h5ad_server.data
        indices_chunk = h5ad_server.indices
        indptr_chunk = h5ad_server.indptr

        r0 = h5ad_server.base_row
        i0 = indptr_chunk[0]
        raw_in_bounds = []
        raw_out_bounds = []
        for r_idx in range(len(indptr_chunk)-1):
            old_row = r_idx + r0
            new_row = self._old_row_to_new_row[old_row]
            if new_row < self._row_chunk[0] or new_row >= self._row_chunk[1]:
                continue
            out_idx0 = self._new_row_to_idx[new_row]-self.idx_min
            out_idx1 = self._new_row_to_idx[new_row+1]-self.idx_min
            data_i0 = indptr_chunk[r_idx] + i0
            data_i1 = indptr_chunk[r_idx+1] + i0
            if out_idx1 != out_idx0:
                raw_in_bounds.append((data_i0, data_i1))
                raw_out_bounds.append((out_idx0, out_idx1))
            else:
                assert data_i0 == data_i1

        n_raw_bounds = len(raw_in_bounds)
        print("merging bounds")
        _t0 = time.time()
        (in_bound_list,
         out_bound_list) = _merge_bounds(raw_in_bounds, raw_out_bounds)
        dur = (time.time()-_t0)/3600.0
        print(f"_merge_bounds took {dur:.2e} hrs -- {len(out_bound_list)} discrete chunks "
              f"from {n_raw_bounds}")

        data_buffer = None
        indices_buffer = None

        t_process = 0.0
        t_write = 0.0

        for in_bounds, out_bounds in zip(in_bound_list, out_bound_list):
            _t0 = time.time()
            min_dex_arr = np.array([o[0] for o in out_bounds])
            sorted_dex = np.argsort(min_dex_arr)
            sorted_in_bounds = [in_bounds[ii] for ii in sorted_dex]
            sorted_out_bounds = [out_bounds[ii] for ii in sorted_dex]
            n_data = sorted_out_bounds[-1][1]-sorted_out_bounds[0][0]
            if data_buffer is None or len(data_buffer) < n_data:
                data_buffer = np.zeros(n_data, dtype=data_chunk.dtype)
                indices_buffer = np.zeros(n_data, dtype=int)
            i0 = 0
            for in_chunk in sorted_in_bounds:
                delta = in_chunk[1]-in_chunk[0]
                i1 = i0+delta
                data_buffer[i0:i1] = data_chunk[in_chunk[0]: in_chunk[1]]
                indices_buffer[i0:i1] = indices_chunk[in_chunk[0]: in_chunk[1]]
                i0 = i1

            out_idx0 = sorted_out_bounds[0][0]
            out_idx1 = sorted_out_bounds[-1][1]
            t_process += time.time()-_t0
            _t0 = time.time()
            file_handle['data'][out_idx0:out_idx1] = data_buffer[:n_data]
            file_handle['indices'][out_idx0:out_idx1] = indices_buffer[:n_data]
            t_write += time.time()-_t0
        print(f"t_process {t_process/3600.0:.2e} t_write {t_write/3600.0:.2e} hrs "
              f"-- {len(in_bound_list)} chunks")

        self.t_write += time.time()-t0


def _merge_bounds(
        raw_in_bounds,
        raw_out_bounds):

    start_to_idx = dict()
    end_to_idx = dict()
    for ii, b in enumerate(raw_out_bounds):
        s = b[0]
        if s in start_to_idx:
            raise RuntimeError(
                f"Cannot merge bounds; duplicate start {s}\n"
                f"{raw_out_bounds}")
        start_to_idx[s] = ii
        e = b[1]
        if e in end_to_idx:
            raise RuntimeError(
                f"Cannot merge bounds; duplicate end {e}\n"
                f"{raw_out_bounds}")
        end_to_idx[e] = ii

    parent_to_child = dict()
    parents = set()
    children = set()
    for ii, b in enumerate(raw_out_bounds):
        if b[1] in start_to_idx:
            this_child = start_to_idx[b[1]]
            parent_to_child[ii] = this_child
            children.add(this_child)
            parents.add(ii)

    all_idx = set(range(len(raw_out_bounds)))
    root_set = all_idx-children
    leaf_set = all_idx-parents
    singleton_set = all_idx-parents-children
    logged_set = set()

    in_bounds = []
    out_bounds = []
    for root in root_set:
        this_in = []
        this_out = []
        this_in.append(raw_in_bounds[root])
        this_out.append(raw_out_bounds[root])
        logged_set.add(root)
        while root in parent_to_child:
            root = parent_to_child[root]
            this_in.append(raw_in_bounds[root])
            this_out.append(raw_out_bounds[root])
            logged_set.add(root)
        in_bounds.append(this_in)
        out_bounds.append(this_out)

    for idx in range(len(raw_in_bounds)):
        if idx in logged_set:
            continue
        in_bounds.append([raw_in_bounds[idx]])
        out_bounds.append([raw_out_bounds[idx]])
    return in_bounds, out_bounds
