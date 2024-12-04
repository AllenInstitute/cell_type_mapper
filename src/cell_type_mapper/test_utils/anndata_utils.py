import anndata
import h5py

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
