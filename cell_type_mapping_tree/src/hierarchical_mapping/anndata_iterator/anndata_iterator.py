import anndata
import h5py
import numpy as np


class AnnDataIterator(object):
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

        data = anndata.read_h5ad(h5ad_path, backed='r')
        self._chunk_iterator = data.chunked_X(
            chunk_size=row_chunk_size)

    def __iter__(self):
        return self

    def __next__(self):
        result = next(self._chunk_iterator)[0]
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        return result
