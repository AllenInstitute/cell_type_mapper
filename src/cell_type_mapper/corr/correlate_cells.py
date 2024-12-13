import h5py
import json
import multiprocessing
import numpy as np
import time
import warnings

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.utils.utils import (
    print_timing)

from cell_type_mapper.anndata_iterator.anndata_iterator import (
    AnnDataRowIterator)

from cell_type_mapper.utils.distance_utils import (
    correlation_dot,
    correlation_nearest_neighbors)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list)

from cell_type_mapper.corr.utils import (
    match_genes)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


def corrmap_cells(
        query_path,
        precomputed_path,
        marker_gene_list=None,
        rows_at_a_time=100000,
        n_processors=4,
        tmp_dir=None,
        query_normalization='raw',
        log=None,
        max_gb=10):
    """
    query_path is the path to the h5ad file containing the query cells

    precomputed_path is the path to the precomputed stats file

    marker_gene_list is an optional list of marker gene identifiers

    Returns list of dicts like
        {'assignment': cluster_name,
         'confidence': confidence_value,
         'cell_id': cell_id}
    """

    # if rows_at_a_time is larger than n_query_cells/n_processors,
    # lower it
    obs = read_df_from_h5ad(query_path, 'obs')
    n_cells = len(obs.index.values)
    max_chunk_size = np.ceil(n_cells/n_processors).astype(int)
    if rows_at_a_time > max_chunk_size:
        rows_at_a_time = max_chunk_size

    all_query_genes = _get_query_genes(query_path)

    (reference_profiles,
     cluster_to_row_lookup,
     gene_idx,
     cell_id_list,
     row_iterator) = _prep_data(
                         query_path=query_path,
                         precomputed_path=precomputed_path,
                         marker_gene_list=marker_gene_list,
                         rows_at_a_time=rows_at_a_time,
                         tmp_dir=tmp_dir,
                         log=log,
                         max_gb=max_gb)

    row_to_cluster_lookup = {
        cluster_to_row_lookup[n]: n
        for n in cluster_to_row_lookup.keys()}

    mgr = multiprocessing.Manager()
    output_list = mgr.list()
    output_lock = mgr.Lock()
    process_list = []

    for query_chunk_meta in row_iterator:

        query_chunk = query_chunk_meta[0]
        r0 = query_chunk_meta[1]
        r1 = query_chunk_meta[2]

        query_chunk = CellByGeneMatrix(
            data=query_chunk,
            gene_identifiers=all_query_genes,
            normalization=query_normalization)

        # must convert to log2CPM before downsampling by
        # genes because downsampling the genes affects
        # the converstion to CPM
        if query_chunk.normalization != 'log2CPM':
            query_chunk.to_log2CPM_in_place()

        query_chunk.downsample_genes_in_place(
            selected_genes=gene_idx['names'])

        cell_id_chunk = cell_id_list[r0:r1]
        p = multiprocessing.Process(
                target=_corrmap_worker,
                kwargs={
                    'reference_profiles': reference_profiles,
                    'query_chunk': query_chunk,
                    'row_to_cluster_lookup': row_to_cluster_lookup,
                    'cell_id_chunk': cell_id_chunk,
                    'output_list': output_list,
                    'output_lock': output_lock})
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)
    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    return list(output_list)


def _corrmap_worker(
        reference_profiles,
        query_chunk,
        row_to_cluster_lookup,
        cell_id_chunk,
        output_list,
        output_lock):

    result = []

    (cluster_idx,
     correlation_values) = correlation_nearest_neighbors(
                 baseline_array=reference_profiles,
                 query_array=query_chunk.data,
                 return_correlation=True)
    for idx, corr, cell_id in zip(cluster_idx,
                                  correlation_values,
                                  cell_id_chunk):
        this = {'cell_id': cell_id,
                'cluster': {
                    'assignment': row_to_cluster_lookup[idx],
                    'confidence': corr}}
        result.append(this)

    with output_lock:
        for element in result:
            output_list.append(element)


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
    output_key = 'correlation'

    (reference_profiles,
     cluster_to_row_lookup,
     gene_idx,
     _,
     row_iterator) = _prep_data(
                         query_path=query_path,
                         precomputed_path=precomputed_path,
                         marker_gene_list=marker_gene_list,
                         rows_at_a_time=rows_at_a_time,
                         tmp_dir=tmp_dir)

    n_query_rows = row_iterator.n_rows
    n_clusters = reference_profiles.shape[0]

    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset(
            'cluster_to_col',
            data=json.dumps(cluster_to_row_lookup).encode('utf-8'))

        out_file.create_dataset(
            output_key,
            shape=(n_query_rows, n_clusters),
            dtype=float,
            chunks=(min(1000, n_query_rows),
                    min(1000, n_clusters)))

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
                    chunk_unit='cells',
                    unit=None)

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)


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


def _get_query_cell_id(query_path):
    obs = read_df_from_h5ad(query_path, 'obs')
    return list(obs.index.values)


def _prep_data(
        precomputed_path,
        query_path,
        marker_gene_list,
        rows_at_a_time,
        tmp_dir,
        log=None,
        max_gb=10):
    """
    Prepare data for flat mapping

    Parameters
    ----------
    precomputed_path
        path to precomputed stats file
    query_path
        path to query h5ad file
    marker_gene_list
        optional list of marker genes to use
    rows_at_a_time
        int indicating how many cells to map with
        a single worker
    tmp_dir
        optional directory for writing temporary data products
    log
        optional CommandLog for recording messages when run
        with CLI

    Returns
    -------
    reference_profiles
        an (n_clusters, n_genes) array
    cluster_to_row_lookup
        dict mapping cluster name to row index
    cell_id:
        list of cell ids
    gene_idx
        dict

        'reference' -> indices of reference
        genes being used (reference_profiles has already been
        downsampled to these genes)

        'query' -> indices of query genes being used

        'names' -> list of gene names

    row_iterator
        n iterator over chunks of query cells
    """

    query_genes = _get_query_genes(query_path)

    if marker_gene_list is not None:
        marker_gene_set = set(marker_gene_list)
        query_gene_set = set(query_genes)

        if len(query_gene_set.intersection(marker_gene_set)) == 0:
            raise RuntimeError(
                f"No marker genes appeared in query dataset {query_path}")

    (reference_profiles,
     reference_genes,
     cluster_to_row_lookup) = _get_reference_profiles(precomputed_path)

    if marker_gene_list is not None:
        if len(set(reference_genes).intersection(marker_gene_set)) == 0:
            raise RuntimeError(
                "No marker genes in reference dataset\n"
                f"precomputed file path: {precomputed_path}")

    gene_idx = match_genes(
                    reference_gene_names=reference_genes,
                    query_gene_names=query_genes,
                    marker_gene_names=marker_gene_list)

    if len(gene_idx['reference']) == 0:
        raise RuntimeError(
             "Cannot map celltypes; no gene overlaps")

    if marker_gene_list is not None:
        if len(gene_idx['reference']) < len(marker_gene_list):

            reference_gene_set = set(reference_genes)
            not_in_ref = marker_gene_set-reference_gene_set
            not_in_query = marker_gene_set-query_gene_set
            all_missing = not_in_ref.union(not_in_query)

            msg = f"The following {len(all_missing)} "
            msg += "marker genes are being skipped.\n"
            if len(not_in_query) > 0:
                not_in_query = list(not_in_query)
                not_in_query.sort()
                msg += f"These {len(not_in_query)} genes were not present "
                msg += "in the query dataset:\n"
                msg += f"{not_in_query}\n"
            if len(not_in_ref) > 0:
                not_in_ref = list(not_in_ref)
                not_in_ref.sort()
                msg += f"These {len(not_in_ref)} were not present "
                msg += "in the reference dataset:\n"
                msg += f"{not_in_ref}"
            if log is not None:
                log.warn(msg)
            else:
                warnings.warn(msg)

    reference_profiles = reference_profiles[:, gene_idx['reference']]

    cell_id_list = _get_query_cell_id(query_path)

    row_iterator = AnnDataRowIterator(
        h5ad_path=query_path,
        row_chunk_size=rows_at_a_time,
        tmp_dir=tmp_dir,
        log=log,
        max_gb=max_gb)

    return (reference_profiles,
            cluster_to_row_lookup,
            gene_idx,
            cell_id_list,
            row_iterator)


def _get_reference_profiles(
        precomputed_path):
    """
    Read in path to precomputed stats file

    Return:
        reference_profiles
            an (n_clusters, n_genes) array
        reference_genes
            a list
        cluster_to_row_lookup
            dict mapping cluster name to row in reference_profiles
    """

    with h5py.File(precomputed_path, 'r') as reference_file:
        reference_genes = json.loads(
            reference_file['col_names'][()].decode('utf-8'))

        reference_profiles = reference_file['sum'][()]
        reference_profiles = reference_profiles.transpose()
        reference_profiles = reference_profiles/reference_file['n_cells'][()]
        reference_profiles = reference_profiles.transpose()

        cluster_to_row_lookup = json.loads(
            reference_file['cluster_to_row'][()].decode('utf-8'))

    return (reference_profiles,
            reference_genes,
            cluster_to_row_lookup)
