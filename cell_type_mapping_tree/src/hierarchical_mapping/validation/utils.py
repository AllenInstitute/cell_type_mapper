import anndata
import gc
import h5py
import numpy as np
import pathlib
import tempfile

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)


def round_x_to_integers(
        h5ad_path,
        tmp_dir=None):
    """
    If X matrix in h5ad file is not integers, round to nearest integers,
    saving the new data to the h5ad file in question.

    tmp_dir is a directory where the new data can be written before being
    copied over
    """
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir,
            prefix='round_x_to_integers_'))

    tmp_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            prefix='data_as_int_',
            suffix='.h5'))

    with h5py.File(h5ad_path, 'r') as src:
        attrs = dict(src['X'].attrs)
    encoding_type = attrs['encoding-type']
    if encoding_type == 'array':
        _round_dense_x_to_integers(
            h5ad_path=h5ad_path,
            tmp_path=tmp_path)
    elif 'csr' in encoding_type or 'csc' in encoding_type:
        _round_sparse_x_to_integers(
            h5ad_path=h5ad_path,
            tmp_path=tmp_path)
    else:
        raise RuntimeError(
            "Do not know how to handle encoding-type "
            f"{encoding_type}")

    _clean_up(tmp_dir)


def get_minmax_x_from_h5ad(
        h5ad_path):
    """
    Find the minimum and maximum value of the X matrix in an h5ad file.
    """
    with h5py.File(h5ad_path, 'r') as in_file:
        attrs = dict(in_file['X'].attrs)
        if 'encoding-type' not in attrs:
            pass
        elif attrs['encoding-type'] == 'array':
            return _get_minmax_from_dense(in_file['X'])
        elif 'csr' in attrs['encoding-type'] \
                or 'csc' in attrs['encoding-type']:
            return _get_minmax_from_sparse(in_file['X'])
        else:
            pass

    return _get_minmax_x_using_anndata(h5ad_path)


def _get_minmax_x_using_anndata(
        h5ad_path,
        rows_at_a_time=10000):
    """
    If you cannot intuit how X is encoded in the h5ad file, just use
    anndata's API

    Returns
    -------
    (min_val, max_val)
    """
    max_val = None
    min_val = None
    a_data = anndata.read_h5ad(h5ad_path, backed='r')
    n_rows = a_data.X.shape[0]
    for r0 in range(0, n_rows, rows_at_a_time):
        r1 = min(n_rows, r0+rows_at_a_time)
        chunk = a_data.chunk_X[np.arange(r0, r1)]
        this_max = chunk.max()
        if max_val is None or this_max > max_val:
            max_val = this_max
        this_min = chunk.min()
        if min_val is None or this_min < min_val:
            min_val = this_min

    del a_data
    gc.collect()

    return (min_val, max_val)


def _get_minmax_from_dense(x_dataset):
    """
    Get the minimum and maximum values from the X array if it is dense

    Parameters
    ----------
    x_dataset:
        The HDF5 dataset containing X

    Returns
    -------
    (min_val, max_val)
    """
    if x_dataset.chunks is None:
        x = x_dataset[()]
        return (x.min(), x.max())
    min_val = None
    max_val = None
    chunk_size = x_dataset.chunks
    for r0 in range(0, x_dataset.shape[0], chunk_size[0]):
        r1 = min(x_dataset.shape[0], r0+chunk_size[0])
        for c0 in range(0, x_dataset.shape[1], chunk_size[1]):
            c1 = min(x_dataset.shape[1], c0+chunk_size[1])
            chunk = x_dataset[r0:r1, c0:c1]
            chunk_min = chunk.min()
            chunk_max = chunk.max()
            if min_val is None or chunk_min < min_val:
                min_val = chunk_min
            if max_val is None or chunk_max > max_val:
                max_val = chunk_max
    return (min_val, max_val)


def _get_minmax_from_sparse(x_grp):
    """
    Get the minimum and maximum values from the X array if it is sparse

    Parameters
    ----------
    x_grp:
        The HDF5 group containing X

    Returns
    -------
    (min_val, max_val)
    """
    data_dataset = x_grp['data']
    if data_dataset.chunks is None:
        data_dataset = data_dataset[()]
        return (data_dataset.min(), data_dataset.max())

    min_val = None
    max_val = None

    chunk_size = data_dataset.chunks
    n_el = data_dataset.shape[0]

    for i0 in range(0, n_el, chunk_size[0]):
        i1 = min(n_el, i0+chunk_size[0])
        chunk = data_dataset[i0:i1]
        chunk_min = chunk.min()
        chunk_max = chunk.max()
        if min_val is None or chunk_min < min_val:
            min_val = chunk_min
        if max_val is None or chunk_max > max_val:
            max_val = chunk_max
    return (min_val, max_val)


def _round_dense_x_to_integers(
        h5ad_path,
        tmp_path):
    with h5py.File(h5ad_path, 'r') as src:
        with h5py.File(tmp_path, 'w') as dst:
            data = src['X']
            chunk_size = data.chunks

            dst.create_dataset(
                'data',
                shape=data.shape,
                chunks=chunk_size,
                dtype=int)

            if chunk_size is None:
                chunk_size = data.shape

            for r0 in range(0, data.shape[0], chunk_size[0]):
                r1 = min(data.shape[0], r0+chunk_size[0])
                for c0 in range(0, data.shape[1], chunk_size[1]):
                    c1 = min(data.shape[1], c0+chunk_size[1])
                    chunk = data[r0:r1, c0:c1]
                    chunk = np.round(chunk).astype(int)
                    dst['data'][r0:r1, c0:c1] = chunk

    with h5py.File(h5ad_path, 'a') as dst:
        with h5py.File(tmp_path, 'r') as src:
            attrs = dict(dst['X'].attrs)
            del dst['X']
            dataset = dst.create_dataset(
                'X',
                data=src['data'][()],
                dtype=src['data'].dtype,
                chunks=src['data'].chunks)
            for k in attrs:
                dataset.attrs.create(
                    name=k,
                    data=attrs[k])


def _round_sparse_x_to_integers(
        h5ad_path,
        tmp_path):
    with h5py.File(h5ad_path, 'r') as src:
        with h5py.File(tmp_path, 'w') as dst:
            data = src['X/data']
            chunk_size = data.chunks

            dst.create_dataset(
                'data',
                shape=data.shape,
                chunks=chunk_size,
                dtype=int)

            if chunk_size is None:
                chunk_size = data.shape

            for i0 in range(0, data.shape[0], chunk_size[0]):
                i1 = min(data.shape[0], i0+chunk_size[0])
                chunk = data[i0:i1]
                chunk = np.round(chunk).astype(int)
                dst['data'][i0:i1] = chunk

    with h5py.File(h5ad_path, 'a') as dst:
        with h5py.File(tmp_path, 'r') as src:
            del dst['X/data']
            dst.create_dataset(
                'X/data',
                data=src['data'][()],
                dtype=src['data'].dtype,
                chunks=src['data'].chunks)
