import h5py
import json
import numpy as np
import pathlib
import shutil
import tempfile

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)

from hierarchical_mapping.diff_exp.sparse_markers import (
    add_sparse_markers_to_h5)


def thin_marker_file(
        marker_file_path,
        thinned_marker_file_path,
        max_bytes=6*1024**3,
        tmp_dir=None):
    """
    Remove all rows (genes) from marker_file_path that are not
    markers for any cluster pairs.

    Parameters
    ----------
    marker_file_path:
        Path to the original (un-thinned) marker file
    thinned_marker_file_path:
        File to be written
    max_bytes:
        Maximum number of bytes to load at once
    tmp_dir:
        Optional fast temp dir to copy input file into before thinning
    """

    marker_file_path = pathlib.Path(
        marker_file_path)
    thinned_marker_file_path = pathlib.Path(
        thinned_marker_file_path)
    src = str(marker_file_path.resolve().absolute())
    dst = str(thinned_marker_file_path.resolve().absolute())
    if src == dst:
        raise RuntimeError(
            "marker_file_path == thinned_marker_file_path\n"
            "Will not run in this mode; they must be different")

    if tmp_dir is not None:
        tmp_dir = pathlib.Path(
            tempfile.mkdtemp(dir=tmp_dir, prefix='thinning_'))
        new_path = mkstemp_clean(dir=tmp_dir, suffix='.h5')
        shutil.copy(src=marker_file_path, dst=new_path)
        marker_file_path = pathlib.Path(new_path)
        tmp_thinned_path = mkstemp_clean(
            dir=tmp_dir,
            prefix='thin_',
            suffix='.h5')
    else:
        tmp_thinned_path = thinned_marker_file_path

    with h5py.File(tmp_thinned_path, "w") as dst:
        with h5py.File(marker_file_path, "r") as src:
            dst.create_dataset('n_pairs', data=src['n_pairs'][()])
            dst.create_dataset('pair_to_idx', data=src['pair_to_idx'][()])
            base_shape = src['markers/data'].shape

    rows_to_keep = find_nonzero_rows(
        path=marker_file_path,
        row0=0,
        row1=base_shape[0],
        n_cols=base_shape[1],
        max_bytes=max_bytes)

    rows_to_keep = np.array(rows_to_keep)
    rows_to_keep = np.sort(rows_to_keep)
    n_keep = len(rows_to_keep)

    rows_at_a_time = max(1000, max_bytes//base_shape[1])

    with h5py.File(tmp_thinned_path, "a") as dst:
        with h5py.File(marker_file_path, "r") as src:
            full_gene_names = json.loads(src['gene_names'][()].decode('utf-8'))

            dst.create_dataset(
                'full_gene_names',
                data=src['gene_names'][()])

            dst.create_dataset(
                'gene_names',
                data=json.dumps(
                    [full_gene_names[ii]
                     for ii in rows_to_keep]).encode('utf-8'))

            dst.create_dataset(
                "markers/data",
                dtype=np.uint8,
                shape=(n_keep, base_shape[1]),
                chunks=(min(n_keep, 1000), min(base_shape[1], 1000)))

            dst.create_dataset(
                "up_regulated/data",
                dtype=np.uint8,
                shape=(n_keep, base_shape[1]),
                chunks=(min(n_keep, 1000), min(base_shape[1], 1000)))

            for i0 in range(0, base_shape[0], rows_at_a_time):
                i1 = min(base_shape[0], i0+rows_at_a_time)

                valid = np.logical_and(
                    rows_to_keep >= i0,
                    rows_to_keep < i1)
                these_rows_to_keep = rows_to_keep[valid]-i0
                map_to = np.where(valid)[0]
                if valid.sum() == 0:
                    continue
                for k in ('markers/data', 'up_regulated/data'):
                    chunk = src[k][i0:i1, :]
                    chunk = chunk[these_rows_to_keep, :]
                    dst[k][map_to, :] = chunk

    add_sparse_markers_to_h5(tmp_thinned_path)

    if tmp_thinned_path != thinned_marker_file_path:
        print(f"moving {tmp_thinned_path} to {thinned_marker_file_path}")
        shutil.move(
            src=tmp_thinned_path,
            dst=thinned_marker_file_path)

    if tmp_dir is not None:
        print(f"cleaning {tmp_dir}")
        _clean_up(tmp_dir)


def find_nonzero_rows(
        path,
        row0,
        row1,
        n_cols,
        max_bytes):
    """
    Scane the markers/data array in the file at path. Log any rows between
    row0 and row1 that are nonzero (i.e. that are markers for any pairs).

    Record them in output_list.

    n_cols is the number of columns in the array (used to set how many
    rows to load at a time)

    Return list of non-zero rows
    """
    non_zero_rows = []
    rows_at_a_time = max(1, max_bytes//n_cols)
    for r0 in range(row0, row1, rows_at_a_time):
        r1 = min(row1, r0+rows_at_a_time)
        with h5py.File(path, 'r', swmr=True) as src:
            chunk = src['markers/data'][r0:r1, :]
        chunk = chunk.sum(axis=1)
        non_zero_rows += list(np.where(chunk > 0)[0] + r0)

    return non_zero_rows
