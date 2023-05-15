import anndata
import h5py
import numpy as np
import pathlib
import tempfile
import time

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.utils.sparse_utils import (
    load_csc)


class AnnDataRowIterator(object):
    """
    A class to efficiently iterate over the rows of an anndata
    file. If the anndata file is CSC, it will, as a first step,
    write the data out to a tempfile in CSR (or dense) format
    for more rapid iteration over rows.
    """

    def __init__(
            self,
            h5ad_path,
            row_chunk_size,
            tmp_dir=None):

        self.tmp_dir = None
        h5ad_path = pathlib.Path(h5ad_path)
        if not h5ad_path.is_file():
            raise RuntimeError(
                f"{h5ad_path} is not a file")

        with h5py.File(h5ad_path, 'r') as in_file:
            attrs = dict(in_file['X'].attrs)
        if attrs['encoding-type'].startswith('csr'):
            self._initialize_anndata_iterator(
                h5ad_path=h5ad_path,
                row_chunk_size=row_chunk_size)
        elif attrs['encoding-type'].startswith('array'):
            self._initialize_anndata_iterator(
                h5ad_path=h5ad_path,
                row_chunk_size=row_chunk_size)
        elif attrs['encoding-type'].startswith('csc'):
            self._initialize_as_csc(
                h5ad_path=h5ad_path,
                row_chunk_size=row_chunk_size,
                tmp_dir=tmp_dir)
        else:
            raise RuntimeError(
                "AnnDataRowIterator cannot handle encoding\n"
                f"{attrs}")

    def __del__(self):
        if self.tmp_dir is not None:
            _clean_up(self.tmp_dir)

    def __iter__(self):
        return self

    def __next__(self):
        result = next(self._chunk_iterator)
        if self._iterator_type == 'anndata':
            r0 = result[1]
            r1 = result[2]
            result = result[0]
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            result = (result, r0, r1)
        return result

    def _initialize_anndata_iterator(self, h5ad_path, row_chunk_size):
        self._iterator_type = 'anndata'
        data = anndata.read_h5ad(h5ad_path, backed='r')
        self.n_rows = data.X.shape[0]
        self._chunk_iterator = data.chunked_X(
            chunk_size=row_chunk_size)

    def _initialize_as_csc(
            self,
            h5ad_path,
            row_chunk_size,
            tmp_dir=None):
        self._iterator_type = 'anndata'
        if tmp_dir is None:
            self._initialize_anndata_iterator(
                h5ad_path=h5ad_path,
                row_chunk_size=row_chunk_size)
        else:
            with h5py.File(h5ad_path, 'r') as src:
                attrs = dict(src['X'].attrs)
                csc_dtype = src['X/data'].dtype

            if 'shape' not in attrs:
                self._initialize_anndata_iterator(
                    h5ad_path=h5ad_path,
                    row_chunk_size=row_chunk_size)
                return

            self.tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
            self.tmp_path = pathlib.Path(
                mkstemp_clean(
                    dir=self.tmp_dir,
                    prefix=f"{h5ad_path.name}_as_dense_",
                    suffix=".h5"))

            t0 = time.time()
            print(f"transcribing {h5ad_path} to {self.tmp_path} "
                  "as a dense array")

            array_shape = attrs['shape']
            self.n_rows = array_shape[0]
            one_mb = 1024**2
            one_gb = 1024**3
            rows_per_chunk = np.ceil(5*one_mb/(4*array_shape[1])).astype(int)
            rows_per_chunk = max(1, rows_per_chunk)
            cols_per_chunk = np.ceil(20*one_gb/(4*array_shape[0])).astype(int)
            cols_per_chunk = max(1, cols_per_chunk)

            rows_per_chunk = min(array_shape[0], rows_per_chunk)
            cols_per_chunk = min(array_shape[1], cols_per_chunk)

            print(f"rows_per_chunk {rows_per_chunk} cols {cols_per_chunk}")

            with h5py.File(self.tmp_path, 'w') as dst:
                dst_data = dst.create_dataset(
                                'data',
                                dtype=csc_dtype,
                                shape=array_shape,
                                chunks=(rows_per_chunk,
                                        array_shape[1]))

                print("created empty dataset")
                with h5py.File(h5ad_path, 'r') as src:
                    for c0 in range(0, array_shape[1], cols_per_chunk):
                        c1 = min(array_shape[1], c0+cols_per_chunk)
                        print(f"    col {c0}:{c1}")
                        chunk = load_csc(
                            col_spec=(c0, c1),
                            n_rows=array_shape[0],
                            data=src['X/data'],
                            indices=src['X/indices'],
                            indptr=src['X/indptr'])
                        dst_data[:, c0:c1] = chunk

            self._iterator_type = 'dense'
            self._chunk_iterator = DenseIterator(
                h5_path=self.tmp_path,
                row_chunk_size=row_chunk_size)

            duration = time.time()-t0
            print(f"transcription took {duration:.2e} seconds")


class DenseIterator(object):

    def __init__(self, h5_path, row_chunk_size):
        self.h5_path = h5_path
        self.h5_handle = None
        self.row_chunk_size = row_chunk_size
        self.r0 = 0
        self.h5_handle = h5py.File(h5_path, 'r', swmr=True)
        self.n_rows = self.h5_handle['data'].shape[0]

    def __next__(self):
        if self.r0 >= self.n_rows:
            if self.h5_handle is not None:
                self.h5_handle.close()
                self.h5_handle = None
            raise StopIteration
        r1 = min(self.n_rows, self.r0+self.row_chunk_size)
        chunk = self.h5_handle['data'][self.r0:r1, :]
        old_r0 = self.r0
        self.r0 = r1
        return (chunk, old_r0, r1)

    def __del__(self):
        if self.h5_handle is not None:
            self.h5_handle.close()
            self.h5_handle = None
