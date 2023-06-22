from anndata._io.specs import read_elem
from anndata._io.specs import write_elem
import h5py


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
