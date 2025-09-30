import h5py
import json

import cell_type_mapper.utils.utils as ctm_utils


def write_hann_metadata(
        metadata,
        log,
        log_path,
        hdf5_output_path,
        cloud_safe):
    """
    Append metadata to a HANN run output

    Parameters
    ----------
    metadata:
        dict containing metadata for the mapping run
    log:
        CommandLog object associated with the mapping run
    log_path:
        path to the text file where log should be written
    hdf5_output_path:
        path to HDF5 file with output
    cloud_safe:
        boolean indicating whether or not we are in cloud_safe
        mode
    """
    if log_path is not None:
        log.write_log(log_path, cloud_safe=cloud_safe)

    with h5py.File(hdf5_output_path, "a") as dst:
        dst.create_dataset(
            "metadata",
            data=json.dumps(
                ctm_utils.clean_for_json(metadata)
            ).encode('utf-8')
        )
