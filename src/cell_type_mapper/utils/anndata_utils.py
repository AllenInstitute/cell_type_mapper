import anndata
from anndata._io.specs import read_elem
from anndata._io.specs import write_elem
import h5py
import numpy as np
import pandas as pd
import warnings

from cell_type_mapper.utils.sparse_utils import (
    amalgamate_sparse_array)


def read_df_from_h5ad(h5ad_path, df_name):
    """
    Read the dataframe df_name (probably 'obs' or 'var')
    from the h5ad file at h5ad_path
    """
    with h5py.File(h5ad_path, 'r') as src:
        return read_elem(src[df_name])


def write_df_to_h5ad(h5ad_path, df_name, df_value):
    """
    Write the data in df_value to the element df_name in the
    specified h5ad_path
    """
    with h5py.File(h5ad_path, 'a') as dst:
        try:
            write_elem(dst, key=df_name, val=df_value)
        except TypeError:
            write_elem(dst, k=df_name, elem=df_value)


def read_uns_from_h5ad(h5ad_path):
    """
    Read the unstructured metadata dict
    from the h5ad file at h5ad_path
    """
    with h5py.File(h5ad_path, 'r') as src:
        if 'uns' not in src:
            return dict()
        return read_elem(src['uns'])


def write_uns_to_h5ad(h5ad_path, uns_value):
    """
    Write the data in un_value to the element uns in the
    specified h5ad_path
    """
    with h5py.File(h5ad_path, 'a') as dst:
        try:
            write_elem(dst, key='uns', val=uns_value)
        except TypeError:
            write_elem(dst, k='uns', elem=uns_value)


def does_obsm_have_key(h5ad_path, obsm_key):
    """
    Return a boolean assessing whether or not obsm has
    the specified key
    """
    with h5py.File(h5ad_path, 'r') as src:
        k_list = set(src['obsm'].keys())
    return obsm_key in k_list


def append_to_obsm(
        h5ad_path,
        obsm_key,
        obsm_value,
        clobber=False):
    """
    Add some data to the 'obsm' element of an H5AD file.

    Parameters
    ----------
    h5ad_path:
        Path to the H5AD file
    obsm_key:
        The key in obsm to which the new data will be assigned
    obsm_value:
        The data to be written
    clobber:
        If False, raise an error if obsm_key is already in
        obsm.
    """
    if isinstance(obsm_value, pd.DataFrame):
        obs = read_df_from_h5ad(h5ad_path, df_name='obs')
        obs_keys = list(obs.index.values)
        these_keys = list(obsm_value.index.values)
        if not set(obs_keys) == set(these_keys):
            raise RuntimeError(
                "Cannot write dataframe to obsm; index values "
                "are not the same as the index values in obs.")

        if obs_keys != these_keys:
            obsm_value = obsm_value.loc[obs_keys]

    with h5py.File(h5ad_path, 'a') as dst:
        obsm = read_elem(dst['obsm'])
        if not isinstance(obsm, dict):
            raise RuntimeError(
                f"'obsm' is not a dict; it is a {type(obsm)}\n"
                "Unclear how to proceed")
        if not clobber:
            if obsm_key in obsm:
                raise RuntimeError(
                    f"{obsm_key} already in obsm. Cannot write "
                    f"data to {h5ad_path}")

        obsm[obsm_key] = obsm_value

        try:
            write_elem(dst, key='obsm', val=obsm)
        except TypeError:
            write_elem(dst, k='obsm', elem=obsm)


def copy_layer_to_x(
        original_h5ad_path,
        new_h5ad_path,
        layer):
    """
    Copy the data in original_h5ad_path over to new_h5ad_path.

    Copy over only obs, var and the specified layer, moving the
    layer into 'X'

    Note: specified layer can be 'X'. This is apparently a faster
    way to copy over an h5ad file than using shutil.copy.
    """
    if layer == 'X':
        layer_key = 'X'
    else:
        layer_key = f'layers/{layer}'
    obs = read_df_from_h5ad(original_h5ad_path, 'obs')
    var = read_df_from_h5ad(original_h5ad_path, 'var')
    output = anndata.AnnData(obs=obs, var=var)
    output.write_h5ad(new_h5ad_path)
    with h5py.File(original_h5ad_path, 'r') as src:
        attrs = dict(src[layer_key].attrs)

    if 'encoding-type' in attrs:
        encoding_type = attrs['encoding-type']
    else:
        warnings.warn(
            f"{original_h5ad_path}['{layer_key}'] had no "
            "encoding-type listed; will assume it is a "
            "dense array")
        encoding_type = 'array'

    if encoding_type == 'array':
        _copy_layer_to_x_dense(
            original_h5ad_path=original_h5ad_path,
            new_h5ad_path=new_h5ad_path,
            layer_key=layer_key)
    elif 'csr' in encoding_type or 'csc' in encoding_type:
        _copy_layer_to_x_sparse(
            original_h5ad_path=original_h5ad_path,
            new_h5ad_path=new_h5ad_path,
            layer_key=layer_key)
    else:
        raise RuntimeError(
            "unclear how to copy layer with attrs "
            f"{attrs}")


def _copy_layer_to_x_dense(
        original_h5ad_path,
        new_h5ad_path,
        layer_key):
    with h5py.File(original_h5ad_path) as src:
        data = src[layer_key]
        attrs = dict(src[layer_key].attrs)
        chunks = data.chunks
        if chunks is None:
            row_chunk = min(10000, data.shape[0]//10)
            if row_chunk == 0:
                row_chunk = data.shape[0]
            # col_chunk = min(10000, data.shape[1]//10)
            col_chunk = data.shape[1]
            if col_chunk == 0:
                col_chunk = data.shape[1]

            chunks = (row_chunk, col_chunk)

        with h5py.File(new_h5ad_path, 'a') as dst:
            if 'X' in dst:
                del dst['X']

            dst_dataset = dst.create_dataset(
                'X',
                shape=data.shape,
                chunks=chunks,
                dtype=data.dtype)

            written_attrs = set()
            for k in attrs:
                dst_dataset.attrs.create(
                    name=k,
                    data=attrs[k])
                written_attrs.add(k)

            if 'encoding-type' not in written_attrs:
                dst_dataset.attrs.create(
                    name='encoding-type',
                    data='array')

            if chunks is None:
                chunks = data.shape
            for r0 in range(0, data.shape[0], chunks[0]):
                r1 = min(data.shape[0], r0+chunks[0])
                for c0 in range(0, data.shape[1], chunks[1]):
                    c1 = min(data.shape[1], c0+chunks[1])
                    dst_dataset[r0:r1, c0:c1] = data[r0:r1, c0:c1]


def _copy_layer_to_x_sparse(
        original_h5ad_path,
        new_h5ad_path,
        layer_key):
    with h5py.File(original_h5ad_path) as src:
        src_grp = src[layer_key]
        attrs = dict(src_grp.attrs)
        with h5py.File(new_h5ad_path, 'a') as dst:
            if 'X' in dst:
                del dst['X']

            dst_grp = dst.create_group('X')

            for k in attrs:
                dst_grp.attrs.create(
                    name=k,
                    data=attrs[k])

            for el in ('indptr', 'indices', 'data'):
                src_dataset = src_grp[el]
                dtype = src_dataset.dtype
                chunks = src_dataset.chunks
                dst_grp.create_dataset(
                    el,
                    shape=src_dataset.shape,
                    chunks=chunks,
                    dtype=dtype)
                if chunks is None:
                    dst_grp[el] = src_dataset[()]
                else:
                    for i0 in range(0, src_dataset.shape[0], chunks[0]):
                        i1 = min(src_dataset.shape[0], i0+chunks[0])
                        dst_grp[el][i0:i1] = src_dataset[i0:i1]


def amalgamate_h5ad(
        src_rows,
        dst_path,
        dst_obs,
        dst_var,
        verbose=False,
        tmp_dir=None):

    """
    Take rows (or columns for csc matrices) from different
    sparse arrays stored in different h5ad files and combine
    them into a single sparse array in a single h5ad file.

    Parameters
    ----------
    src_rows:
        Ordered list of dicts. Each dict is
        {
            'path': /path/to/src/file
            'rows': [ordered list of rows/columns from that file]
        }

    dst_path:
        Path of file to be written

    dst_obs:
        The obs dataframe for the final file

    dst_var:
        The var dataframe for the final file

    verbose:
        If True, issue print statements indicating the
        status of the copy

    tmp_dir:
        Directory where temporary files will be written
    """
    # check that all files are csr matrices
    for src_element in src_rows:
        src_path = src_element['path']
        with h5py.File(src_path, 'r') as src:
            attrs = dict(src['X'].attrs)
        if attrs['encoding-type'] != 'csr_matrix':
            raise RuntimeError(
                f"{src_path} is {attrs}\nnot 'csr_matrix'")

    a_data = anndata.AnnData(obs=dst_obs, var=dst_var)
    a_data.write_h5ad(dst_path)

    amalgamate_sparse_array(
        src_rows=src_rows,
        dst_path=dst_path,
        sparse_grp='X',
        verbose=verbose,
        tmp_dir=tmp_dir)

    with h5py.File(dst_path, 'a') as dst:
        x_handle = dst['X']
        x_handle.attrs.create(
            name='encoding-type', data='csr_matrix')
        x_handle.attrs.create(
            name='encoding-version', data='0.1.0')
        x_handle.attrs.create(
            name='shape', data=np.array([len(dst_obs), len(dst_var)]))
