import h5py
import json
import multiprocessing
import numpy as np
import time
from scipy.spatial.distance import cdist as scipy_cdist

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list)

from hierarchical_mapping.corr.utils import (
    match_genes)

from hierarchical_mapping.utils.sparse_utils import (
    load_csr)


def correlate_cells(
        query_path,
        precomputed_path,
        output_path,
        gb_at_a_time=16,
        n_processors=4):
    """
    query_path is the path to the h5ad file containing the query cells

    precomputed_path is the path to the precomputed stats file

    output_path is the path to the HDF5 file that will be written
    correlating cells with clusters
    """

    output_key = 'correlation'

    with h5py.File(query_path, 'r') as query_file:
        query_genes = [s.decode('utf-8')
                       for s in query_file['var/_index'][()]]

        query_indptr = query_file['X/indptr'][()]
        n_query_rows = len(query_indptr)-1
        n_query_cols = len(query_genes)

    with h5py.File(precomputed_path, 'r') as reference_file:
        reference_genes = json.loads(
            reference_file['col_names'][()].decode('utf-8'))
        gene_idx = match_genes(
                        reference_gene_names=reference_genes,
                        query_gene_names=query_genes)

        if len(gene_idx['reference']) == 0:
            raise RuntimeError(
                 "Cannot map celltypes; no gene overlaps")

        reference_profiles = reference_file['sum'][()]
        reference_profiles = reference_profiles[:, gene_idx['reference']]
        reference_profiles = reference_profiles.transpose()/reference_file['n_cells'][()]
        reference_profiles = reference_profiles.transpose()

        n_clusters = reference_profiles.shape[0]

        rows_at_a_time = np.round(gb_at_a_time*1024**3/(8*n_clusters)).astype(int)
        rows_at_a_time = max(1, rows_at_a_time)
        print(f"rows at a time {rows_at_a_time:.2e}")

        with h5py.File(output_path, 'w') as out_file:
            out_file.create_dataset(
                'cluster_to_col',
                data=reference_file['cluster_to_row'][()])

            out_file.create_dataset(
                output_key,
                shape=(n_query_rows, n_clusters),
                dtype=float,
                chunks=(min(1000, n_query_rows),
                        min(1000, n_clusters)))

    print("starting correlation")
    t0 = time.time()
    row_ct = 0
    process_list = []

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()

    for r0 in range(0, n_query_rows, rows_at_a_time):
        r1 = min(n_query_rows, r0+rows_at_a_time)
        with h5py.File(query_path, 'r') as query_file:
            query_chunk = load_csr(
                    row_spec=(r0, r1),
                    n_cols=n_query_cols,
                    data=query_file['X/data'],
                    indices=query_file['X/indices'],
                    indptr=query_indptr)

        query_chunk = query_chunk[:, gene_idx['query']]

        p = multiprocessing.Process(
                target=_correlate_chunk,
                kwargs={
                    "reference_profiles": reference_profiles,
                    "query_chunk": query_chunk,
                    "row_spec": (r0, r1),
                    "output_path": output_path,
                    "output_key": output_key,
                    "output_lock": output_lock})
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            n0 = len(process_list)
            process_list = winnow_process_list(process_list)
            if len(process_list) < n0:
                row_ct += (n0-len(process_list))*rows_at_a_time
                print_timing(
                    t0=t0,
                    i_chunk=row_ct,
                    tot_chunks=n_query_rows,
                    unit='hr')

    for p in process_list:
        p.join()

    print("all done")


def _correlate_chunk(
        reference_profiles,
        query_chunk,
        row_spec,
        output_path,
        output_key,
        output_lock):
    """
    Correlate a chunk of cells against the mean cluster profiles.

    Parameters
    ----------
    reference_profiles
        A (n_clusters, n_genes) np.ndarray
    query_chunk
        A (n_query_cells, n_genes) np.ndarray
    row_spec
        (row_min, row_max) tuple indicating which cells
        in the whole query dataset these are
    output_path
        Path to the HDF5 file where the correlation data
        is to be stored
    output_key
        Name of the dataset where the correlation
        data is to be stored
    output_lock
        Multiprocessing lock to make sure more than one
        worker does not try to write to output at once

    Returns
    -------
    None
        Data is written to file at output_path

    Notes
    -----
    reference_profiles and query_chunk have already been
    downselected to the shared set of genes
    """
    corr = 1.0-scipy_cdist(query_chunk,
                           reference_profiles,
                           metric='correlation')
    with output_lock:
        with h5py.File(output_path, 'a') as out_file:
            out_file[output_key][row_spec[0]: row_spec[1], :] = corr
