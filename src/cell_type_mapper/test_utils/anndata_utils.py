import anndata
import gzip
import h5py
import numpy as np
import pathlib

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    read_uns_from_h5ad
)


def create_h5ad_without_encoding_type(
        src_path,
        dst_path):
    """
    Take the h5ad file at src_path and copy it to
    dst_path intentionally leaving out all of the useful
    metadata about how the matrices are encoded (dense,
    CSC, or CSR)

    This is meant to enable unit tests that exercise the
    case where, for whatever reason, that metadata is missing
    from the h5ad file.

    Note: this function will only copies
        obs
        var
        X
        layers/
        raw/
        uns
    """
    obs = read_df_from_h5ad(src_path, df_name='obs')
    var = read_df_from_h5ad(src_path, df_name='var')
    uns = read_uns_from_h5ad(src_path)
    a_data = anndata.AnnData(obs=obs, var=var, uns=uns)
    a_data.write_h5ad(dst_path)

    with h5py.File(src_path, 'r') as src:
        with h5py.File(dst_path, 'a') as dst:
            if 'layers' in dst:
                del dst['layers']
            if 'X' in dst:
                del dst['X']
            dst_layers = dst.create_group('layers')

            _copy_array_no_encoding_type(
                src_handle=src['X'],
                dst_grp=dst,
                dst_handle='X'
            )

            if 'raw' in src:
                if 'raw' in dst:
                    del dst['raw']
                dst_raw = dst.create_group('raw')
                for specific_layer in src['raw'].keys():
                    _copy_array_no_encoding_type(
                        src_handle=src['raw'][specific_layer],
                        dst_grp=dst_raw,
                        dst_handle=specific_layer
                    )

            for specific_layer in src['layers'].keys():
                _copy_array_no_encoding_type(
                    src_handle=src['layers'][specific_layer],
                    dst_grp=dst_layers,
                    dst_handle=specific_layer
                )


def _copy_array_no_encoding_type(
        src_handle,
        dst_grp,
        dst_handle):
    """
    Parameters
    ----------
    src_handle:
        handle pointing to the source data
    dst_grp:
        handle pointing to (the existing) handle for the group
        in dst where the data will be written
    dst_handle:
        name of the dataset/group under dst_grp to be written
    """
    if isinstance(src_handle, h5py.Dataset):
        dst_grp.create_dataset(
            dst_handle,
            data=src_handle[()]
        )
    else:
        data_grp = dst_grp.create_group(dst_handle)
        for dataset in src_handle.keys():
            data_grp.create_dataset(
                dataset,
                data=src_handle[dataset][()]
            )


def write_anndata_x_to_csv(
        anndata_obj,
        dst_path,
        cell_label_header=False,
        cell_label_type='string'):
    """
    Write the data in anndata_obj to a csv file at dst_path,
    using the X layer as the matrix.

    (This is for testing our validation layer's ability to
    convert CSVs to h5ad files)

    Valid entries for cell_label_type:
        'string', 'numerical', 'big_numerical' or None
    """

    if cell_label_header:
        if cell_label_type is None:
            raise RuntimeError("this makes no sense")

    dst_path = pathlib.Path(dst_path)
    if dst_path.name.endswith('gz'):
        open_fn = gzip.open
        is_gzip = True
    else:
        open_fn = open
        is_gzip = False

    # find an offset for cell_label_type == 'numerical'
    # that would break the results of a unit test
    numerical_offset = 0
    if cell_label_type == 'numerical':
        masked_x = np.ma.masked_array(
            anndata_obj.X,
            mask=(anndata_obj.X == 0)
        )
        mu = np.mean(masked_x, axis=1)
        std = np.std(masked_x, axis=1, ddof=1)
        numerical_offset = np.round(np.min((mu+2*std))).astype(int)
        extremal = np.min(mu+3*std)

        # make sure we would not trigger the "is the first
        # column just too big" filter on numerical cell labels
        assert numerical_offset < extremal

    with open_fn(dst_path, 'w') as dst:
        header = ''
        if cell_label_type is not None:
            if cell_label_header:
                header += 'a_cell_label'
        for i_gene, gene_label in enumerate(anndata_obj.var.index.values):
            if i_gene > 0 or cell_label_type is not None:
                header += ','
            header += f'{gene_label}'
        header += '\n'
        if is_gzip:
            header = header.encode('utf-8')
        dst.write(header)
        for i_row, cell_label in enumerate(anndata_obj.obs.index.values):
            line = ''
            if cell_label_type is not None:
                if cell_label_type == 'string':
                    line = f'{cell_label}'
                elif cell_label_type == 'numerical':
                    line = f'{numerical_offset + i_row}'
                elif cell_label_type == 'big_numerical':
                    line = f'{1000000+800*i_row}'

            row = anndata_obj.X[i_row, :]
            for i_value, value in enumerate(row):
                if i_value > 0 or cell_label_type is not None:
                    line += ','
                line += f'{value:.6f}'
            line += '\n'
            if is_gzip:
                line = line.encode('utf-8')
            dst.write(line)
