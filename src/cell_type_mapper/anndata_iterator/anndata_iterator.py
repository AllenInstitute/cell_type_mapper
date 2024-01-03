import h5py
import os
import pathlib
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.csc_to_csr import (
    csc_to_csr_on_disk)

from cell_type_mapper.utils.sparse_utils import (
    load_csr)


class h5_handler_manager():
    def __init__(self, h5_path, mode='r', keepopen=True):
        self.keepopen = keepopen
        self.h5_path = h5_path
        self.mode = mode
        self.h5_handle = None
        if keepopen:
            self.h5_handle = h5py.File(h5_path,
                                       mode,
                                       swmr=True)

    def __enter__(self):
        if not self.keepopen:
            self.h5_handle = h5py.File(
                self.h5_path,
                self.mode,
                swmr=True)
        return self.h5_handle

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.keepopen:
            self.h5_handle.close()

    def close(self):
        if self.keepopen:
            self.h5_handle.close()
            self.h5_handle = None


class AnnDataRowIterator(object):
    """
    A class to efficiently iterate over the rows of an anndata
    file. If the anndata file is CSC, it will, as a first step,
    write the data out to a tempfile in CSR (or dense) format
    for more rapid iteration over rows.

    Parameters
    ----------
    h5ad_path:
        Path to the h5ad file over whose rows we are iterating
    row_chunk_size:
        Number of rows to deliver per chunk
    tmp_dir:
        Optional scratch directory. This is where a hypothetical
        CSC file will be written as a CSR file. If None, the
        CSC file will be iterated over using anndata's infrastructure
        (which can be very slow)
    log:
        an optional CommandLog for tracking warnings during CLI runs
    max_gb:
        maximum number of gigabytes to use (approximate) when converting
        CSC matrix to CSR (if necessary)
    """

    def __init__(
            self,
            h5ad_path,
            row_chunk_size,
            tmp_dir=None,
            log=None,
            max_gb=10,
            keep_open=True):

        self.log = log
        self.tmp_dir = None
        self.max_gb = max_gb
        h5ad_path = pathlib.Path(h5ad_path)
        if not h5ad_path.is_file():
            raise RuntimeError(
                f"{h5ad_path} is not a file")

        with h5py.File(h5ad_path, 'r', swmr=True) as in_file:
            attrs = dict(in_file['X'].attrs)
            array_shape = None
            encoding_type = ''
            if 'shape' in attrs:
                array_shape = attrs['shape']
            if 'encoding-type' in attrs:
                encoding_type = attrs['encoding-type']

        if encoding_type.startswith('csr') and array_shape is not None:
            self._iterator_type = 'CSRRow'
            self.n_rows = array_shape[0]
            self._chunk_iterator = CSRRowIterator(
                h5_path=h5ad_path,
                row_chunk_size=row_chunk_size,
                array_shape=array_shape,
                h5_group='X',
                keep_open=keep_open)
        elif encoding_type.startswith('csc'):
            self._initialize_as_csc(
                h5ad_path=h5ad_path,
                row_chunk_size=row_chunk_size,
                tmp_dir=tmp_dir,
                keep_open=keep_open)
        elif encoding_type.startswith('array'):
            self._iterator_type = "dense"
            with h5py.File(h5ad_path, "r", swmr=True) as src:
                array_shape = src['X'].shape
            self.n_rows = array_shape[0]
            self._chunk_iterator = DenseArrayRowIterator(
                  h5_path=h5ad_path,
                  row_chunk_size=row_chunk_size,
                  array_shape=array_shape,
                  keep_open=keep_open)
        else:
            raise RuntimeError(
                "Do not know how to iterate over anndata "
                f"with attrs\n{attrs}")

    def __del__(self):
        if self.tmp_dir is not None:
            _clean_up(self.tmp_dir)

    def __iter__(self):
        return self

    def __getitem__(self, x):
        return self._chunk_iterator[x]

    def __next__(self):
        """
        Return the next chunk of rows.

        Actually return a tuple
        (row_chunk, r0, r1)
        where r0 and r1 are the indices of the slice of rows
        (i.e. row_chunk is data[r0:r1, :])
        """
        result = next(self._chunk_iterator)
        return result

    def get_chunk(self, r0, r1):
        """
        Returns the tuple (data[r0:r1, :], r0, r1)
        """
        return self._chunk_iterator.get_chunk(
            r0=r0, r1=r1)

    def _initialize_as_csc(
            self,
            h5ad_path,
            row_chunk_size,
            tmp_dir=None,
            keep_open=True):
        """
        Initialize iterator for CSC data. If possible,
        write out data to scratch space as CSR matrix
        and initialize iterator over that.

        Parameters
        ----------
        h5ad_path:
            Path to h5ad file whose rows we are iterating over
        row_chunk_size:
            Number of rows to deliver per chunk
        tmp_dir:
            scratch dir in which to write the CSR form of the data
            (if None, no CSR data will be written and we will just
            use anndata.chunked_X to iterate over the data)
        keep_open:
            boolean indicating whether or not to leave the h5 handle
            open (should be false when using cuda)
        """
        write_as_csr = True
        self.tmp_dir = tempfile.mkdtemp(
            dir=tmp_dir,
            prefix='anndata_iterator_')

        # is there space in self.tmp_dir to write out
        # the file as a CSR file?
        dir_stats = os.statvfs(self.tmp_dir)
        free_bytes = dir_stats.f_bavail*dir_stats.f_bsize
        file_stats = os.stat(h5ad_path)
        file_size_bytes = file_stats.st_size
        fudge_factor = 1.1  # just in case

        if free_bytes < fudge_factor*file_size_bytes:
            write_as_csr = False
        else:
            with h5py.File(h5ad_path, 'r', swmr=True) as src:
                attrs = dict(src['X'].attrs)

            if 'shape' not in attrs:
                write_as_csr = False

        if not write_as_csr:
            raise RuntimeError(
                "Cannot write data as CSR\n"
                f"free_bytes {free_bytes}; file size {file_size_bytes}\n"
                f"attrs:\n{attrs}")
        else:
            self.tmp_path = pathlib.Path(
                mkstemp_clean(
                    dir=self.tmp_dir,
                    prefix=f"{h5ad_path.name}_as_csr_",
                    suffix=".h5"))

            t0 = time.time()
            msg = f"transcribing {h5ad_path} to {self.tmp_path} "
            msg += "as a CSR array"
            if self.log is not None:
                self.log.warn(msg)
            else:
                print(msg)

            array_shape = attrs['shape']
            self.n_rows = array_shape[0]
            with h5py.File(h5ad_path, 'r', swmr=True) as src:
                csc_to_csr_on_disk(
                    csc_group=src['X'],
                    csr_path=self.tmp_path,
                    array_shape=array_shape,
                    max_gb=0.8*self.max_gb)

            self._iterator_type = 'CSRRow'
            self._chunk_iterator = CSRRowIterator(
                h5_path=self.tmp_path,
                row_chunk_size=row_chunk_size,
                array_shape=array_shape,
                keep_open=keep_open)

            duration = time.time()-t0
            if self.log is not None:
                self.log.benchmark(
                    msg=f"transcribing {h5ad_path} to CSR",
                    duration=duration)
            else:
                msg = f"transcription to CSR took {duration:.2e} seconds"
                print(msg)


class CSRRowIterator(object):
    """
    Class to iterate over a CSR matrix using h5py to directly
    access the data (rather than anndata, which can load unnecessary
    data into memory)

    Parameters
    ----------
    h5_path:
        Path to HDF5 file containing CSR matrix data
    row_chunk_size:
        Number of rows to return with each chunk
    array_shape:
        Shape of the array we are iterating over
    h5_group:
        Optional group in the HDF5 file where you will find
        'data', 'indices' and 'indptr'
    """

    def __init__(
            self,
            h5_path,
            row_chunk_size,
            array_shape,
            h5_group=None,
            keep_open=True):

        self.h5_path = h5_path
        self.h5_handle = None
        self.row_chunk_size = row_chunk_size
        self.r0 = 0
        self.n_rows = array_shape[0]
        self.n_cols = array_shape[1]

        self.h5_handler = h5_handler_manager(h5_path, keepopen=keep_open)

        if h5_group is None:
            self.data_key = 'data'
            self.indices_key = 'indices'
            self.indptr_key = 'indptr'
        else:
            self.data_key = f'{h5_group}/data'
            self.indices_key = f'{h5_group}/indices'
            self.indptr_key = f'{h5_group}/indptr'

    def __del__(self):
        if self.h5_handle is not None:
            self.h5_handler.close()

    def __next__(self):
        """
        Actually return a tuple

        (row_chunk, r0, r1)

        where r0 and r1 are the indices of the slice of rows
        (i.e. row_chunk is data[r0:r1, :])
        """
        if self.r0 >= self.n_rows:
            if self.h5_handle is not None:
                self.h5_handle = None
            raise StopIteration
        r1 = min(self.n_rows, self.r0+self.row_chunk_size)
        chunk = self.get_chunk(r0=self.r0, r1=r1)
        self.r0 = r1
        return chunk

    def get_chunk(self, r0, r1):
        """
        Returns the tuple (data[r0:r1, :], r0, r1)
        """
        with self.h5_handler as h5_handle:
            chunk = load_csr(
                row_spec=(r0, r1),
                n_cols=self.n_cols,
                data=h5_handle[self.data_key],
                indices=h5_handle[self.indices_key],
                indptr=h5_handle[self.indptr_key])
        return (chunk, r0, r1)

    def __getitem__(self, r0):
        if isinstance(r0, list):
            r1 = r0[-1] + 1
            r0 = r0[0]
        else:
            r1 = r0 + 1

        with self.h5_handler as h5_handle:
            chunk = load_csr(
                row_spec=(r0, r1),
                n_cols=self.n_cols,
                data=h5_handle[self.data_key],
                indices=h5_handle[self.indices_key],
                indptr=h5_handle[self.indptr_key])

        return (chunk, r0, r1)


class DenseArrayRowIterator(object):
    """
    Class to iterate over a dense array using h5py to directly
    access the data (rather than anndata, which can load unnecessary
    data into memory)

    Parameters
    ----------
    h5_path:
        Path to HDF5 file containing CSR matrix data
    row_chunk_size:
        Number of rows to return with each chunk
    array_shape:
        Shape of the array we are iterating over
    h5_group:
        Optional group in the HDF5 file where you will find
        'data', 'indices' and 'indptr'
    """

    def __init__(
            self,
            h5_path,
            row_chunk_size,
            array_shape,
            keep_open=True):

        self.h5_path = h5_path
        self.h5_handle = None
        self.row_chunk_size = row_chunk_size
        self.r0 = 0
        self.n_rows = array_shape[0]
        self.n_cols = array_shape[1]

        self.h5_handler = h5_handler_manager(h5_path, keepopen=keep_open)
        self.data_key = 'X'

    def __del__(self):
        if self.h5_handle is not None:
            self.h5_handler.close()

    def __next__(self):
        """
        Actually return a tuple

        (row_chunk, r0, r1)

        where r0 and r1 are the indices of the slice of rows
        (i.e. row_chunk is data[r0:r1, :])
        """
        if self.r0 >= self.n_rows:
            if self.h5_handle is not None:
                self.h5_handle = None
            raise StopIteration
        r1 = min(self.n_rows, self.r0+self.row_chunk_size)
        chunk = self.get_chunk(r0=self.r0, r1=r1)
        self.r0 = r1
        return chunk

    def get_chunk(self, r0, r1):
        """
        Returns the tuple (data[r0:r1, :], r0, r1)
        """
        with self.h5_handler as h5_handle:
            chunk = h5_handle[self.data_key][r0:r1, :]
        return (chunk, r0, r1)

    def __getitem__(self, r0):
        if isinstance(r0, list):
            r1 = r0[-1] + 1
            r0 = r0[0]
        else:
            r1 = r0 + 1

        with self.h5_handler as h5_handle:
            chunk = h5_handle[self.data_key][r0:r1, :]

        return (chunk, r0, r1)
