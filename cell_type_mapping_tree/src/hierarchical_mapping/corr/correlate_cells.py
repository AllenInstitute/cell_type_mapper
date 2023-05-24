import h5py
import json
import multiprocessing
import time

from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad)

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

from hierarchical_mapping.utils.distance_utils import (
    correlation_dot)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list)

from hierarchical_mapping.corr.utils import (
    match_genes)


def correlate_cells(
        query_path,
        precomputed_path,
        output_path,
        marker_gene_list=None,
        rows_at_a_time=100000,
        n_processors=4,
        tmp_dir=None):
    """
    query_path is the path to the h5ad file containing the query cells

    precomputed_path is the path to the precomputed stats file

    marker_gene_list is an optional list of marker gene identifiers

    output_path is the path to the HDF5 file that will be written
    correlating cells with clusters
    """
    global_t0 = time.time()
    output_key = 'correlation'

    query_genes = _get_query_genes(query_path)

    if marker_gene_list is not None:
        marker_gene_set = set(marker_gene_list)
        query_genes = [q
                       for q in query_genes
                       if q in marker_gene_set]

        if len(query_genes) == 0:
            raise RuntimeError(
                f"No marker genes appeared in query file {query_path}")

    row_iterator = AnnDataRowIterator(
        h5ad_path=query_path,
        row_chunk_size=rows_at_a_time,
        tmp_dir=tmp_dir)

    n_query_rows = row_iterator.n_rows

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
        reference_profiles = reference_profiles.transpose()
        reference_profiles = reference_profiles/reference_file['n_cells'][()]
        reference_profiles = reference_profiles.transpose()

        n_clusters = reference_profiles.shape[0]

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

    for query_chunk_meta in row_iterator:

        query_chunk = query_chunk_meta[0]
        r0 = query_chunk_meta[1]
        r1 = query_chunk_meta[2]

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

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    duration = (time.time()-global_t0)/3600.0
    print(f"all done -- correlation took {duration:.2e} hrs")


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
    corr = correlation_dot(query_chunk,
                           reference_profiles)
    with output_lock:
        with h5py.File(output_path, 'a') as out_file:
            out_file[output_key][row_spec[0]: row_spec[1], :] = corr


def _get_query_genes(query_path):
    var = read_df_from_h5ad(query_path, 'var')
    return list(var.index.values)
