import h5py
import json
import multiprocessing
import numpy as np
import pathlib
import time


from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list)


def thin_marker_file(
        marker_file_path,
        thinned_marker_file_path,
        n_processors=6,
        max_bytes=6*1024**3):
    """
    Remove all rows (genes) from marker_file_path that are not
    markers for any cluster pairs.

    Parameters
    ----------
    marker_file_path:
        Path to the original (un-thinned) marker file
    thinned_marker_file_path:
        File to be written
    n_processors
        Number of independent workers to spin up
    max_bytes:
        Maximum number of bytes a worker should load at once
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

    with h5py.File(thinned_marker_file_path, "w") as dst:
        with h5py.File(marker_file_path, "r") as src:
            dst.create_dataset('n_pairs', data=src['n_pairs'][()])
            dst.create_dataset('pair_to_idx', data=src['pair_to_idx'][()])
            base_shape = src['markers/data'].shape

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    output_list = mgr.list()

    n_per = np.ceil(base_shape[1]/n_processors).astype(int)
    process_list = []
    for i0 in range(0, base_shape[0], n_per):
        p = multiprocessing.Process(
                target=_find_nonzero_rows,
                kwargs={
                    'path': marker_file_path,
                    'row0': i0,
                    'row1': min(base_shape[0], i0+n_per),
                    'n_cols': base_shape[1],
                    'output_lock': output_lock,
                    'output_list': output_list,
                    'max_bytes': max_bytes})
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

    # because this will raise an exception if one of the
    # jobs exits with a non-zero state
    winnow_process_list(process_list)

    rows_to_keep = np.array(output_list)
    rows_to_keep = np.sort(rows_to_keep)
    n_keep = len(rows_to_keep)

    this_max_bytes = max(1, n_processors//2)*max_bytes
    rows_at_a_time = max(1000, this_max_bytes//base_shape[1])

    with h5py.File(thinned_marker_file_path, "a") as dst:
        with h5py.File(marker_file_path, "r") as src:
            full_gene_names = json.loads(src['gene_names'][()].decode('utf-8'))
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


def _find_nonzero_rows(
        path,
        row0,
        row1,
        n_cols,
        output_lock,
        output_list,
        max_bytes):
    """
    Scane the markers/data array in the file at path. Log any rows between
    row0 and row1 that are nonzero (i.e. that are markers for any pairs).

    Record them in output_list.

    n_cols is the number of columns in the array (used to set how many
    rows to load at a time)
    """
    t0 = time.time()
    these_rows = []
    rows_at_a_time = max(1, max_bytes//n_cols)
    for r0 in range(row0, row1, rows_at_a_time):
        r1 = min(row1, r0+rows_at_a_time)
        with h5py.File(path, 'r', swmr=True) as src:
            chunk = src['markers/data'][r0:r1, :]
        row_sums = chunk.sum(axis=1)
        these_rows += list(np.where(row_sums > 0)[0] + r0)
        dur = time.time()-t0
        print(f"    scanned rows {r0}:{r1} ({r1-r0}) in {dur:.2e} seconds")

    with output_lock:
        for r in these_rows:
            output_list.append(r)
    dur = time.time()-t0
    print(f"scanned rows {row0}:{row1} in {dur:.2e} seconds")
