import h5py
import os


def copy_h5_excluding_data(
        src_path,
        dst_path,
        tmp_dir=None,
        excluded_groups=None,
        excluded_datasets=None):
    """
    Copy HDF5 file from src_path to dst_path excluding
    the groups and datasets listed in excluded_groups
    and excluded_datasets.

    Necessary because using the del operator just deletes
    the name of the dataset in the HDF5 file; not the data
    itself.

    See:
    https://stackoverflow.com/questions/39448961/delete-h5py-datasets-item-but-file-size-double
    """
    if excluded_groups is None:
        excluded_groups = set()
    else:
        excluded_groups = set(excluded_groups)

    if excluded_datasets is None:
        excluded_datasets = set()
    else:
        excluded_datasets = set(excluded_datasets)

    with h5py.File(src_path, 'r') as src:
        with h5py.File(dst_path, 'w') as dst:
            for el in src.keys():
                _copy_h5_element(
                    src_handle=src,
                    dst_handle=dst,
                    current_location=el,
                    excluded_datasets=excluded_datasets,
                    excluded_groups=excluded_groups)


def _copy_h5_element(
        src_handle,
        dst_handle,
        current_location,
        excluded_datasets,
        excluded_groups):

    if isinstance(src_handle[current_location], h5py.Dataset):
        if current_location not in excluded_datasets:
            dst_handle.create_dataset(
                current_location,
                data=src_handle[current_location],
                chunks=src_handle[current_location].chunks)
    else:
        if current_location not in excluded_groups:
            for next_el in src_handle[current_location].keys():
                full_next_el = os.path.join(current_location, next_el)
                _copy_h5_element(
                    src_handle=src_handle,
                    dst_handle=dst_handle,
                    current_location=full_next_el,
                    excluded_datasets=excluded_datasets,
                    excluded_groups=excluded_groups)
