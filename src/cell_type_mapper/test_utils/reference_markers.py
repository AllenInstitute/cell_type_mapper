import h5py
import json
import pathlib
import shutil


def move_precomputed_stats_from_reference_markers(
        reference_marker_path_list,
        tmp_dir):
    """
    Take a list of reference marker files. Copy those files
    to a new directory. Copy their associated precomputed_stats
    files into the same directory. Alter their metadata so that
    the precomputed_stats field points to a non-existent file.
    Return the list of the paths to the doctored reference marker
    files.

    This is for testing how the query marker finder handles cases
    where the absolute path to the precomputed stats file has
    changed, but the precomputed stats file is in the same directory
    as its reference marker file.

    Parameters
    ----------
    reference_marker_path_list:
        List of paths to reference marker files
    tmp_dir:
        pathlib.Path to temporary directory where
        files will be copied

    Returns
    -------
    List of paths to new reference marker files.
    """
    # copy reference_marker files into new locations,
    # changing the entry for precomputed_stats path
    # so that it points to a nonsense file.
    new_ref_marker_list = []
    for src_path in reference_marker_path_list:
        dst_path = tmp_dir / pathlib.Path(src_path).name
        new_ref_marker_list.append(
            str(dst_path.resolve().absolute())
        )
        with h5py.File(dst_path, 'w') as dst:
            with h5py.File(src_path, 'r') as src:
                metadata = json.loads(src['metadata'][()])
                for name in ('gene_names', 'pair_to_idx', 'n_pairs'):
                    dst.create_dataset(
                        name,
                        data=src[name][()])
                for group in ('sparse_by_pair', 'sparse_by_gene'):
                    grp = dst.create_group(group)
                    for name in ('down_gene_idx',
                                 'down_pair_idx',
                                 'up_gene_idx',
                                 'up_pair_idx'):
                        grp.create_dataset(
                            name,
                            data=src[group][name][()])
            precompute_path = pathlib.Path(metadata['precomputed_path'])
            new_precompute = tmp_dir / precompute_path.name
            nonsense_precompute = f'/not/really/a/file/{precompute_path.name}'
            shutil.copy(
                src=precompute_path,
                dst=new_precompute)
            new_metadata = {
                'precomputed_path': nonsense_precompute
            }
            dst.create_dataset(
                'metadata',
                data=json.dumps(new_metadata).encode('utf-8')
            )
    return new_ref_marker_list


def move_precomputed_stats_from_mask_file(
        mask_file_path,
        tmp_dir):
    """
    Take a p-value mask file. Copy it and its linked precomputed_stats file
    to a new directory, altering the metadata of the p-value mask file to
    point to a file that does not exist.

    This is for testing how the query marker finder handles cases
    where the absolute path to the precomputed stats file has
    changed, but the precomputed stats file is in the same directory
    as its p-value mask file.

    Parameters
    ----------
    mask_file_path:
        Path to the p-value mask file
    tmp_dir:
        pathlib.Path to temporary directory where
        files will be copied

    Returns
    -------
    path to the new p-value mask file
    """
    tmp_dir = pathlib.Path(tmp_dir)
    new_mask_path = tmp_dir / pathlib.Path(mask_file_path).name
    with h5py.File(mask_file_path, 'r') as src:
        with h5py.File(new_mask_path, 'w') as dst:
            for name in ('data',
                         'gene_names',
                         'indices',
                         'indptr',
                         'n_pairs',
                         'pair_to_idx'):
                dst.create_dataset(name, data=src[name][()])
            metadata = json.loads(src['metadata'][()])
            src_precompute = metadata['config']['precomputed_stats_path']
            dst_precompute = tmp_dir / 'precompute.h5'
            shutil.copy(src=src_precompute, dst=dst_precompute)
            new_metadata = {
                'config': {
                    'precomputed_stats_path': '/no/such/file/precompute.h5'
                }
            }

            dst.create_dataset(
                'metadata',
                data=json.dumps(new_metadata).encode('utf-8')
            )

    return new_mask_path
