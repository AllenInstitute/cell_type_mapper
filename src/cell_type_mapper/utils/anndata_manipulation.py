import anndata
import h5py
import json
import numpy as np
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up
)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

from cell_type_mapper.utils.anndata_utils import (
    infer_attrs
)


def amalgamate_h5ad(
        src_rows,
        dst_path,
        dst_obs,
        dst_var,
        dst_sparse=True,
        tmp_dir=None,
        compression=True):

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
                     read from 'layers/{some_layer}' (unless '/' is in
                     the layer specification, as in 'raw/X', in which
                     case layer is read directly from that location)
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

    compression:
        A boolean; if True, use gzip with setting 4

    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        _amalgamate_h5ad(
            src_rows=src_rows,
            dst_path=dst_path,
            dst_obs=dst_obs,
            dst_var=dst_var,
            dst_sparse=dst_sparse,
            tmp_dir=tmp_dir,
            compression=compression)
    finally:
        _clean_up(tmp_dir)


def _amalgamate_h5ad(
        src_rows,
        dst_path,
        dst_obs,
        dst_var,
        dst_sparse,
        tmp_dir,
        compression):

    t0 = time.time()

    # check that all source files have data stored in
    # the same dtype
    data_dtype_map = dict()
    for packet in src_rows:

        if packet['layer'] == 'X':
            layer = 'X'
        elif '/' in packet['layer']:
            layer = packet['layer']
        else:
            layer = f'layers/{packet["layer"]}'

        attrs = infer_attrs(
            src_path=packet['path'],
            dataset=layer
        )

        with h5py.File(packet['path'], 'r') as src:
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
    n_packets = len(src_rows)
    ct = 0
    n_print = max(1, n_packets//10)
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

        ct += 1
        if ct % n_print == 0:
            dur = (time.time()-t0)/60.0
            per = dur/ct
            pred = per*n_packets
            remain = pred-dur
            print(
                f"{ct} packets in {dur:.2e} minutes; "
                f"predict {remain:.2e} of {pred:2e} left"
            )

    a_data = anndata.AnnData(obs=dst_obs, var=dst_var)
    a_data.write_h5ad(dst_path)

    print("joining files")

    if dst_sparse:
        amalgamate_csr_to_x(
            src_path_list=tmp_path_list,
            dst_path=dst_path,
            final_shape=(len(dst_obs), len(dst_var)),
            dst_grp='X',
            compression=compression)
    else:
        amalgamate_dense_to_x(
            src_path_list=tmp_path_list,
            dst_path=dst_path,
            final_shape=(len(dst_obs), len(dst_var)),
            dst_grp='X',
            compression=compression)


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
