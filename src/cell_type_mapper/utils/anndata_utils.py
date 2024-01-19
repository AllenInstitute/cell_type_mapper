import anndata
from anndata._io.specs import read_elem
from anndata._io.specs import write_elem
import h5py
import json
import numpy as np
import pandas as pd
import tempfile
import warnings

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)


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


def update_uns(h5ad_path, new_uns, clobber=False):
    """
    Extend the uns element in the specified h5ad file using
    the dict specified in new_uns.

    If clobber is True, overwrite any keys in the original uns
    that exist in new_uns.

    Otherwise, raise an exception of there are duplicate keys.
    """
    uns = read_uns_from_h5ad(h5ad_path)
    if not clobber:
        new_keys = set(new_uns.keys())
        old_keys = set(uns.keys())
        duplicates = new_keys.intersection(old_keys)
        if len(duplicates) > 0:
            duplicates = list(duplicates)
            duplicates.sort()
            msg = (
                "Cannot update uns. The following keys already exist:\n"
                f"{duplicates}"
            )
            raise RuntimeError(msg)
    uns.update(new_uns)
    write_uns_to_h5ad(h5ad_path, uns_value=uns)


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
        dst_sparse=True,
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
            'rows': [ordered list of rows from that file]
            'layer': either 'X' or 'some_layer', in which case data is
                     read from 'layers/some_layer'
        }

    dst_path:
        Path of file to be written

    dst_obs:
        The obs dataframe for the final file

    dst_var:
        The var dataframe for the final file

    dst_sparse:
        A boolean. If True, dst will be written as a CSR
        matrix. Otherwise, it will be written as a dense
        matrix.

    tmp_dir:
        Directory where temporary files will be written
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        _amalgamate_h5ad(
            src_rows=src_rows,
            dst_path=dst_path,
            dst_obs=dst_obs,
            dst_var=dst_var,
            dst_sparse=dst_sparse,
            tmp_dir=tmp_dir)
    finally:
        _clean_up(tmp_dir)


def _amalgamate_h5ad(
        src_rows,
        dst_path,
        dst_obs,
        dst_var,
        dst_sparse,
        tmp_dir):

    # check that all source files have data stored in
    # the same dtype
    data_dtype_map = dict()
    for packet in src_rows:
        if packet['layer'] == 'X':
            layer = 'X'
        else:
            layer = f'layers/{packet["layer"]}'
        with h5py.File(packet['path'], 'r') as src:
            attrs = dict(src[layer].attrs)
            if attrs['encoding-type'] == 'array':
                data_dtype_map[packet['path']] = src[layer].dtype
            else:
                data_dtype_map[packet['path']] = src[f'{layer}/data'].dtype
    if len(set(data_dtype_map.values())) > 1:
        to_output = {
            k: str(data_dtype_map[k])
            for k in data_dtype_map
        }
        raise RuntimeError(
            "Cannot merge h5ad files whose arrays have disparate data types\n"
            f"{json.dumps(to_output, indent=2)}"
        )

    tmp_path_list = []
    for packet in src_rows:

        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            suffix='.h5')

        iterator = AnnDataRowIterator(
            h5ad_path=packet['path'],
            row_chunk_size=1000,
            layer=packet['layer'],
            tmp_dir=tmp_dir,
            log=None,
            max_gb=10)

        row_batch = iterator.get_batch(
            packet['rows'],
            sparse=dst_sparse)

        with h5py.File(tmp_path, 'w') as tmp_dst:
            if dst_sparse:
                tmp_dst.create_dataset(
                    'data', data=row_batch.data)
                tmp_dst.create_dataset(
                    'indices', data=row_batch.indices)
                tmp_dst.create_dataset(
                    'indptr', data=row_batch.indptr)
            else:
                tmp_dst.create_dataset(
                    'data', data=row_batch)

        tmp_path_list.append(tmp_path)

    a_data = anndata.AnnData(obs=dst_obs, var=dst_var)
    a_data.write_h5ad(dst_path)

    if dst_sparse:
        amalgamate_csr_to_x(
            src_path_list=tmp_path_list,
            dst_path=dst_path,
            final_shape=(len(dst_obs), len(dst_var)),
            dst_grp='X')
    else:
        amalgamate_dense_to_x(
            src_path_list=tmp_path_list,
            dst_path=dst_path,
            final_shape=(len(dst_obs), len(dst_var)),
            dst_grp='X')


def amalgamate_csr_to_x(
        src_path_list,
        dst_path,
        final_shape,
        dst_grp='X',
        compression=False):
    """
    Iterate over a list of HDF5 files that store CSR matrices in
    'data', 'indices', and 'indptr'. Combine the matrices into
    a single CSR matrix stored according to the h5ad specification
    in the file specified by dst_path at the group specified
    by dst_grp

    Parameters
    ----------
    src_path_list:
        List of HDF5 files (in order) from which to read CSR data
    dst_path:
        Path to the fil to be written
    final_shape:
        Final shape of the dense array represented by the CSR data
    dst_grp:
        The HDF5 group in which to store the data (e.g. 'X' or
        'layers/my_layer')
    compression:
        A boolean. If True, use gzip with opts=4

    Note
    ----
    Because of whatever is going on here:

    https://github.com/scverse/anndata/issues/1288

    This code will actually only allow you to write data to 'X'.
    """

    if compression:
        compression = 'gzip'
        compression_opts = 4
    else:
        compression = None
        compression_opts = None

    if dst_grp != 'X':
        raise NotImplementedError(
            "amalgamate_csr_to_x cannot write to layers other than 'X'")

    n_valid = 0
    n_indptr = final_shape[0]+1
    data_dtype = None
    indices_max = 0
    for src_path in src_path_list:
        with h5py.File(src_path, 'r') as src:
            n_valid += src['data'].shape[0]
            if data_dtype is None:
                data_dtype = src['data'].dtype
            this_max = src['indices'][()].max()
            if this_max > indices_max:
                indices_max = this_max

    cutoff = np.iinfo(np.int32).max
    if indices_max >= cutoff or n_valid >= cutoff:
        index_dtype = np.int64
    else:
        index_dtype = np.int32

    with h5py.File(dst_path, 'a') as dst:
        grp = dst.create_group(dst_grp)
        grp.attrs.create(
            name='encoding-type', data='csr_matrix')
        grp.attrs.create(
            name='encoding-version', data='0.1.0')
        grp.attrs.create(
            name='shape', data=np.array(final_shape))

        dst_data = grp.create_dataset(
            'data',
            shape=(n_valid,),
            chunks=min(n_valid, 20000),
            dtype=data_dtype,
            compression=compression,
            compression_opts=compression_opts)
        dst_indices = grp.create_dataset(
            'indices',
            shape=(n_valid,),
            chunks=min(n_valid, 20000),
            dtype=index_dtype,
            compression=compression,
            compression_opts=compression_opts)
        dst_indptr = grp.create_dataset(
            'indptr',
            shape=(n_indptr,),
            dtype=index_dtype,
            compression=compression,
            compression_opts=compression_opts)

        indptr0 = 0
        indptr_offset = 0
        data0 = 0
        for src_path in src_path_list:
            with h5py.File(src_path, 'r') as src:
                n_data = src['data'].shape[0]
                dst_data[data0:data0+n_data] = src['data'][()]
                dst_indices[data0:data0+n_data] = src['indices'][()]
                n_rows = src['indptr'].shape[0]-1
                dst_indptr[indptr0:indptr0+n_rows] = (
                    src['indptr'][:-1].astype(index_dtype)
                    + indptr_offset)
                indptr0 += n_rows
                data0 += n_data
                indptr_offset = (src['indptr'][-1].astype(index_dtype)
                                 + indptr_offset)
        dst_indptr[-1] = n_valid


def amalgamate_dense_to_x(
        src_path_list,
        dst_path,
        final_shape,
        dst_grp='X',
        compression=False):
    """
    Iterate over a list of HDF5 files that store dense matrices in
    'data'. Combine the matrices into a single dense matrix stored
    according to the h5ad specification in the file specified by
    dst_path at the group specified by dst_grp.

    Parameters
    ----------
    src_path_list:
        List of HDF5 files (in order) from which to read CSR data
    dst_path:
        Path to the fil to be written
    final_shape:
        Final shape of the dense array represented by the CSR data
    dst_grp:
        The HDF5 group in which to store the data (e.g. 'X' or
        'layers/my_layer')
    compression:
        A boolean. If True, use gzip with opts=4

    Note
    ----
    Because of whatever is going on here:

    https://github.com/scverse/anndata/issues/1288

    This code will actually only allow you to write data to 'X'.
    """

    if compression:
        compression = 'gzip'
        compression_opts = 4
    else:
        compression = None
        compression_opts = None

    if dst_grp != 'X':
        raise NotImplementedError(
            "amalgamate_ense_to_x cannot write to layers other than 'X'")

    n_rows = 0
    n_cols = 0
    data_dtype = None
    for src_path in src_path_list:
        with h5py.File(src_path, 'r') as src:
            shape = src['data'].shape
            n_rows += shape[0]
            if n_cols == 0:
                n_cols = shape[1]
            else:
                if n_cols != shape[1]:
                    raise RuntimeError(
                        "Column mismatch between src files "
                        f"({n_cols} != {shape[1]}")
            if data_dtype is None:
                data_dtype = src['data'].dtype

    found_shape = (n_rows, n_cols)
    if found_shape != final_shape:
        raise RuntimeError(
            f"Expected shape {final_shape}; found{found_shape}")

    with h5py.File(dst_path, 'a') as dst:
        dst_data = dst.create_dataset(
            dst_grp,
            shape=(n_rows, n_cols),
            chunks=(min(n_rows, 1000), min(n_cols, 1000)),
            dtype=data_dtype,
            compression=compression,
            compression_opts=compression_opts)
        dst_data.attrs.create(
            name='encoding-type', data='array')
        dst_data.attrs.create(
            name='encoding-version', data='0.2.0')
        dst_data.attrs.create(
            name='shape', data=np.array(final_shape))

        r0 = 0
        for src_path in src_path_list:
            with h5py.File(src_path, 'r') as src:
                shape = src['data'].shape
                r1 = r0 + shape[0]
                dst_data[r0:r1, :] = src['data'][()]
                r0 = r1
