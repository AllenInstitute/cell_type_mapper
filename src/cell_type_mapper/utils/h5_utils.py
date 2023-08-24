import h5py
import itertools
import numpy as np
import os


def copy_h5_excluding_data(
        src_path,
        dst_path,
        tmp_dir=None,
        excluded_groups=None,
        excluded_datasets=None,
        max_elements=100000):
    """
    Copy HDF5 file from src_path to dst_path excluding
    the groups and datasets listed in excluded_groups
    and excluded_datasets.

    Necessary because using the del operator just deletes
    the name of the dataset in the HDF5 file; not the data
    itself.

    max_elements is the maximum number of scalars to
    copy over in a single chunk

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
                    excluded_groups=excluded_groups,
                    max_elements=max_elements)


def _copy_h5_element(
        src_handle,
        dst_handle,
        current_location,
        excluded_datasets,
        excluded_groups,
        max_elements=100000):
    attrs = dict(src_handle[current_location].attrs)

    if isinstance(src_handle[current_location], h5py.Dataset):
        if current_location not in excluded_datasets:
            chunks = src_handle[current_location].chunks
            if chunks is None:
                dataset = dst_handle.create_dataset(
                    current_location,
                    data=src_handle[current_location],
                    chunks=src_handle[current_location].chunks)
            else:
                src_dataset = src_handle[current_location]
                dst_dataset = dst_handle.create_dataset(
                    current_location,
                    dtype=src_dataset.dtype,
                    shape=src_dataset.shape,
                    chunks=src_dataset.chunks)

                copy_slices = _get_slices_for_copy(
                    data_shape=src_dataset.shape,
                    max_elements=max_elements)

                for this_chunk in itertools.product(*copy_slices):
                    dst_dataset[this_chunk] = src_dataset[this_chunk]

            if len(attrs) > 0:
                for k in attrs:
                    dataset.attrs.create(name=k, data=attrs[k])

    else:
        if current_location not in excluded_groups:
            grp = dst_handle.create_group(current_location)
            if len(attrs) > 0:
                for k in attrs:
                    grp.attrs.create(name=k, data=attrs[k])
            for next_el in src_handle[current_location].keys():
                full_next_el = os.path.join(current_location, next_el)
                _copy_h5_element(
                    src_handle=src_handle,
                    dst_handle=dst_handle,
                    current_location=full_next_el,
                    excluded_datasets=excluded_datasets,
                    excluded_groups=excluded_groups,
                    max_elements=max_elements)


def _get_slices_for_copy(
        data_shape,
        max_elements):
    """
    Returns a list of lists.
    Each sub-list is the list of slices along that dimension
    of data_shape.
    """
    if len(data_shape) > 1:
        per_dim = np.ceil(
            np.power(max_elements, 1.0/len(data_shape))).astype(int)
    else:
        per_dim = max_elements

    n_tot = 1
    actual_slices = []
    for i_dim in range(len(data_shape)):
        this_n = data_shape[i_dim]
        chosen = max(1, min(per_dim, this_n))
        n_tot *= chosen
        these_slices = []
        for i0 in range(0, this_n, chosen):
            i1 = min(i0+chosen, this_n)
            these_slices.append(slice(i0, i1, 1))
        actual_slices.append(these_slices)
    return actual_slices
