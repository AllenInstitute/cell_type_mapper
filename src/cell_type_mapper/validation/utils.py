import h5py
import json
import numpy as np
import pandas as pd
import pathlib
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.anndata_utils import (
    infer_attrs
)

from cell_type_mapper.gene_id.utils import detect_species

from cell_type_mapper.gene_id.gene_id_mapper import (
    GeneIdMapper)


def is_x_integers(
        h5ad_path,
        layer='X'):
    """
    Returns True if the values in X are integers (or, effectively integers).

    Returns False otherwise
    """
    layer_key = _layer_to_layer_key(layer)

    encoding_type = infer_attrs(
        src_path=h5ad_path,
        dataset=layer_key
    )['encoding-type']

    if encoding_type == 'array':
        return _is_dense_x_integers(
            h5ad_path=h5ad_path,
            eps=1.0e-10,
            layer=layer)
    elif 'csr' in encoding_type or 'csc' in encoding_type:
        return _is_sparse_x_integers(
            h5ad_path=h5ad_path,
            eps=1.0e-10,
            layer=layer)
    else:
        raise RuntimeError(
            "Do not know how to handle encoding-type "
            f"{encoding_type}")


def round_x_to_integers(
        h5ad_path,
        tmp_dir=None,
        output_dtype=int):
    """
    If X matrix in h5ad file is not integers, round to nearest integers,
    saving the new data to the h5ad file in question.

    tmp_dir is a directory where the new data can be written before being
    copied over

    output_dtype is the datatype of the array to be saved

    Note: if there is no numeric benefit to casting the data to integers,
    do not do anything.
    """
    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(
            dir=tmp_dir,
            prefix='round_x_to_integers_staging_'))

    tmp_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            prefix='data_as_int_',
            suffix='.h5'))

    attrs = infer_attrs(
        src_path=h5ad_path,
        dataset='X'
    )

    encoding_type = attrs['encoding-type']

    if encoding_type == 'array':
        _round_dense_x_to_integers(
            h5ad_path=h5ad_path,
            tmp_path=tmp_path,
            output_dtype=output_dtype)
    elif 'csr' in encoding_type or 'csc' in encoding_type:
        _round_sparse_x_to_integers(
            h5ad_path=h5ad_path,
            tmp_path=tmp_path,
            output_dtype=output_dtype)
    else:
        raise RuntimeError(
            "Do not know how to handle encoding-type "
            f"{encoding_type}")

    _clean_up(tmp_dir)


def is_data_ge_zero(
        h5ad_path,
        layer='X'):
    """
    Is the specified layer greater than or equal to zero.

    Return a boolean and the minimum value of the data.

    Note: if the data is a uint, the minimum value returned
    will be zero, regardless of what the actual minimum value
    of the data is.
    """
    layer_key = _layer_to_layer_key(layer)
    attrs = infer_attrs(
        src_path=h5ad_path,
        dataset=layer_key
    )
    with h5py.File(h5ad_path, 'r') as in_file:

        if attrs['encoding-type'] == 'array':
            dtype = in_file[layer_key].dtype
        elif 'csr' in attrs['encoding-type'] \
                or 'csc' in attrs['encoding-type']:
            dtype = in_file[f'{layer_key}/data'].dtype
        else:
            raise RuntimeError(
                "Unclear what to make of encoding-typ in attrs:\n"
                f"{attrs}"
            )

        if np.issubdtype(dtype, np.integer):
            iinfo = np.iinfo(dtype)
            if iinfo.min >= 0:
                return True, 0

    minmax = get_minmax_x_from_h5ad(
        h5ad_path=h5ad_path,
        layer=layer)

    if minmax[0] < 0.0:
        return False, minmax[0]

    return True, minmax[0]


def get_minmax_x_from_h5ad(
        h5ad_path,
        layer='X'):
    """
    Find the minimum and maximum value of the X matrix in an h5ad file.
    """

    layer_key = _layer_to_layer_key(layer)
    encoding_type = infer_attrs(
        src_path=h5ad_path,
        dataset=layer_key
    )['encoding-type']

    with h5py.File(h5ad_path, 'r') as in_file:
        if encoding_type == 'array':
            return _get_minmax_from_dense(in_file[layer_key])
        elif 'csr' in encoding_type \
                or 'csc' in encoding_type:
            return _get_minmax_from_sparse(in_file[layer_key])


def map_gene_ids_in_var(
        var_df,
        gene_id_mapper=None,
        log=None):
    """
    Fix the index of the var dataframe to use the preferred gene identifiers
    specified in a GeneIdMapper

    Parameters
    ----------
    var_df:
        original var dataframe
    gene_id_mapper:
        GeneIdMapper containing data needed to map between gene identification
        schemes. If None, infer the mapper based on the species implied by
        the input gene IDs.
    log:
        Optional logger for recording messages.
    Returns
    -------
    If the var dataframe needs to be updated, return the updated
    var dataframe and the number of genes that were unable to be
    mapped.

    If not, return None (and 0)
    """

    gene_id_list = list(var_df.index.values)

    if gene_id_mapper is None:
        species = detect_species(gene_id_list)

        if species is None:
            msg = (
                "Could not find a species for the genes you gave:\n"
                f"First five genes:\n{gene_id_list[:5]}"
            )
            if log is not None:
                log.error(msg)
            else:
                raise RuntimeError(msg)

        if log is not None:
            log.info(f"Mapping genes to {species} genes")

        gene_id_mapper = GeneIdMapper.from_species(
            species=species,
            log=log)

    mapping_output = gene_id_mapper.map_gene_identifiers(gene_id_list)
    new_gene_id_list = mapping_output['mapped_genes']
    if new_gene_id_list == gene_id_list:
        return None, 0

    var_df = var_df.reset_index().to_dict(orient='records')
    idx_key_root = f'{gene_id_mapper.preferred_type}_VALIDATED'
    idx_key = idx_key_root
    ct = 0
    while idx_key in var_df[0]:
        idx_key = f'{idx_key_root}_{ct}'
        ct += 1

    for record, gene_id in zip(var_df, new_gene_id_list):
        record[idx_key] = gene_id

    new_var = pd.DataFrame(var_df).set_index(idx_key)

    return new_var, mapping_output['n_unmapped']


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
    raw_chunk_size = x_dataset.chunks
    x_shape = x_dataset.shape
    ntot = x_shape[0]*x_shape[1]
    nchunk = raw_chunk_size[0]*raw_chunk_size[1]
    chunk_size = (raw_chunk_size[0], raw_chunk_size[1])
    while nchunk < 1000000000 and nchunk < ntot//2:
        chunk_size = (chunk_size[0]*2, chunk_size[1]*2)
        nchunk = chunk_size[0]*chunk_size[1]

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

    ntot = data_dataset.shape[0]

    chunk_size = data_dataset.chunks
    while chunk_size[0] < 1000000000 and chunk_size[0] < ntot//2:
        chunk_size = (chunk_size[0]*2,)

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
        tmp_path,
        output_dtype=int):
    delta = 0.0
    with h5py.File(h5ad_path, 'r') as src:
        with h5py.File(tmp_path, 'w') as dst:
            data = src['X']
            chunk_size = data.chunks

            dst.create_dataset(
                'data',
                shape=data.shape,
                chunks=chunk_size,
                dtype=output_dtype)

            if chunk_size is None:
                chunk_size = data.shape

            for r0 in range(0, data.shape[0], chunk_size[0]):
                r1 = min(data.shape[0], r0+chunk_size[0])
                for c0 in range(0, data.shape[1], chunk_size[1]):
                    c1 = min(data.shape[1], c0+chunk_size[1])
                    chunk = data[r0:r1, c0:c1]
                    rounded_chunk = np.round(chunk)
                    this_delta = np.abs(rounded_chunk-chunk).max()
                    if this_delta > delta:
                        delta = this_delta
                    dst['data'][r0:r1, c0:c1] = rounded_chunk.astype(
                                                    output_dtype)

    # if something changed, actually transcribe the new data
    eps = 1.0e-10
    if delta > eps:
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
        tmp_path,
        output_dtype=int):
    delta = 0.0
    with h5py.File(h5ad_path, 'r') as src:
        with h5py.File(tmp_path, 'w') as dst:
            data = src['X/data']
            chunk_size = data.chunks

            dst.create_dataset(
                'data',
                shape=data.shape,
                chunks=chunk_size,
                dtype=output_dtype)

            if chunk_size is None:
                chunk_size = data.shape

            for i0 in range(0, data.shape[0], chunk_size[0]):
                i1 = min(data.shape[0], i0+chunk_size[0])
                chunk = data[i0:i1]
                rounded_chunk = np.round(chunk)
                this_delta = np.abs(chunk-rounded_chunk).max()
                if this_delta > delta:
                    delta = this_delta
                dst['data'][i0:i1] = rounded_chunk.astype(output_dtype)

    # if something changed, actually transcribe the new data
    eps = 1.0e-10
    if delta > eps:
        with h5py.File(h5ad_path, 'a') as dst:
            with h5py.File(tmp_path, 'r') as src:
                del dst['X/data']
                dst.create_dataset(
                    'X/data',
                    data=src['data'][()],
                    dtype=src['data'].dtype,
                    chunks=src['data'].chunks)


def _is_dense_x_integers(
        h5ad_path,
        eps=1.0e-10,
        layer='X'):
    """
    Returns True if the values in X are integers (or, effectively integers).

    Returns False otherwise

    eps governs how close a float can be to an integer
    and still be called an integer
    """
    layer_key = _layer_to_layer_key(layer)

    with h5py.File(h5ad_path, 'r') as src:
        data = src[layer_key]
        if np.issubdtype(data.dtype, np.integer):
            return True

        chunk_size = data.chunks

        if chunk_size is None:
            chunk_size = data.shape

        for r0 in range(0, data.shape[0], chunk_size[0]):
            r1 = min(data.shape[0], r0+chunk_size[0])
            for c0 in range(0, data.shape[1], chunk_size[1]):
                c1 = min(data.shape[1], c0+chunk_size[1])
                chunk = data[r0:r1, c0:c1]
                rounded_chunk = np.round(chunk)
                this_delta = np.abs(rounded_chunk-chunk).max()
                if this_delta > eps:
                    return False
    return True


def _is_sparse_x_integers(
        h5ad_path,
        eps=1.0e-6,
        layer='X'):
    """
    Returns True if the values in X are integers (or, effectively integers).

    Returns False otherwise

    eps governs how close a float can be to an integer
    and still be called an integer
    """
    layer_key = _layer_to_layer_key(layer)

    with h5py.File(h5ad_path, 'r') as src:
        data = src[f'{layer_key}/data']
        if np.issubdtype(data.dtype, np.integer):
            return True
        chunk_size = data.chunks

        if chunk_size is None:
            chunk_size = data.shape

        for i0 in range(0, data.shape[0], chunk_size[0]):
            i1 = min(data.shape[0], i0+chunk_size[0])
            chunk = data[i0:i1]
            rounded_chunk = np.round(chunk)
            this_delta = np.abs(chunk-rounded_chunk).max()
            if this_delta > eps:
                return False

    return True


def _layer_to_layer_key(layer):
    if layer == 'X':
        layer_key = layer
    else:
        layer_key = f'layers/{layer}'
    return layer_key


def create_uniquely_indexed_df(input_df):
    """
    Take a dataframe. If that dataframe's index has unique values,
    return that same data frame. If it does not, create a dataframe
    with the same data, replacing the index with a string that records
    the original index as well as the row number of the entry.
    """
    index_name = input_df.index.name
    index_values = input_df.index.values
    unq, ct = np.unique(index_values, return_counts=True)

    if len(index_values) == len(unq):
        return input_df

    offenders = [
        unq[ii]
        for ii in range(len(unq))
        if ct[ii] > 1
    ]

    if index_name is None:
        msg = (
            "Index does not have unique values, but index also "
            "is not named. Unclear how to proceed.\n"
            f"Repeated index values are\n{offenders}"
        )
        raise RuntimeError(msg)

    offenders = set(offenders)
    data = input_df.reset_index().to_dict(orient='records')
    for i_row in range(len(data)):
        row = data[i_row]
        if row[index_name] in offenders:
            new_name = {
                index_name: row[index_name],
                'row': i_row
            }
            row[index_name] = json.dumps(new_name)

    new_df = pd.DataFrame(data).set_index(index_name)
    return new_df
