from typing import Union, Dict, List, Any
import time
import h5py
import zarr
import scipy.sparse as scipy_sparse
import numpy as np
import pathlib
import json
import multiprocessing

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.sparse_utils import (
    _load_disjoint_csr)

from hierarchical_mapping.utils.stats_utils import (
    summary_stats_for_chunk)


def precompute_summary_stats(
        data_path: Union[str, pathlib.Path],
        cluster_to_input_row: Dict[str, List[int]],
        n_genes: int,
        output_path: Union[str, pathlib.Path],
        n_processors:int = 6):
    """
    Precompute the summary stats used to identify marker genes

    Parameters
    ----------
    data_path:
        Path to the cell x gene data (stored in zarr as a
        csr sparse array)

    cluster_to_input_row:
        Dict mapping the name of cell clusters to lists
        of the row indexes of cells in those clusters

    n_genes:
        Number of genes in the dataset (not obvious from
        sparse array data)

    output_path:
        Path to the HDF5 file that will contain the lookup
        information for the clusters
    """

    cluster_list = list(cluster_to_input_row)
    cluster_to_output_row = {c: int(ii)
                             for ii, c in enumerate(cluster_list)}
    n_clusters = len(cluster_list)

    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset(
            'cluster_to_row',
            data=json.dumps(cluster_to_output_row).encode('utf-8'))

        out_file.create_dataset('n_cells', shape=(n_clusters,), dtype=int)
        for (k, dt) in (('sum', float), ('sumsq', float),
                        ('gt0', int), ('gt1', int)):
            out_file.create_dataset(k,
                                    shape=(n_clusters, n_genes),
                                    chunks=((max(1, n_clusters//10), n_genes)),
                                    dtype=dt)

    # sub-divide clusters for divsion among worker processes
    worker_division = []
    for ii in range(n_processors):
        worker_division.append(dict())

    n_per = np.ceil(len(cluster_to_input_row)/n_processors).astype(int)

    for ii, cluster in enumerate(cluster_to_input_row.keys()):
        this = cluster_to_input_row[cluster]
        jj = ii // n_per
        worker_division[jj][cluster] = this

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    process_list = []
    for cluster_set in worker_division:
        p = multiprocessing.Process(
                target=_summary_stats_worker,
                kwargs={
                    'data_path': data_path,
                    'cluster_to_input_row': cluster_set,
                    'cluster_to_output_row': cluster_to_output_row,
                    'n_genes': n_genes,
                    'output_path': output_path,
                    'output_lock': output_lock})
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


def _summary_stats_worker(
        data_path: Union[str, pathlib.Path],
        cluster_to_input_row: Dict[str, List[int]],
        cluster_to_output_row: Dict[str, int],
        n_genes: int,
        output_path: Union[str, pathlib.Path],
        output_lock: Any):
    """
    Precompute the summary stats used to identify marker genes

    Parameters
    ----------
    data_path:
        Path to the cell x gene data (stored in zarr as a
        csr sparse array)

    cluster_to_input_row:
        Dict mapping the name of cell clusters to lists
        of the row indexes of cells in those clusters
        (just the clusters to be processed by this worker)

    cluster_to_output_row:
        Dict mapping cluster name to the position in the output
        file where that cluster's data is stored

    n_genes:
        Number of genes in the dataset (not obvious from
        sparse array data)

    output_path:
        Path to the HDF5 file that will contain the lookup
        information for the clusters

    output_lock:
        Lock to prevent multiple workers from writing to the output
        file at once.
    """
    n_clusters = len(cluster_to_input_row)

    results = dict()
    t0 = time.time()
    with zarr.open(data_path, 'r') as data_src:
        for i_cluster, cluster in enumerate(cluster_to_input_row.keys()):
            row_list = cluster_to_input_row[cluster]
            (data,
             indices,
             indptr) = _load_disjoint_csr(
                            row_index_list=row_list,
                            data=data_src['data'],
                            indices=data_src['indices'],
                            indptr=data_src['indptr'])

            csr = scipy_sparse.csr_array(
                        (data, indices, indptr),
                        shape=(len(row_list), n_genes))

            summary_stats = summary_stats_for_chunk(
                                cell_x_gene=csr)

            output_idx = cluster_to_output_row[cluster]
            results[output_idx] = summary_stats

            print_timing(
               t0=t0,
               i_chunk=i_cluster+1,
               tot_chunks=n_clusters,
               unit='min')

        with output_lock:
            with h5py.File(output_path, 'a') as out_file:
                for output_idx in results:
                    summary_stats = results[output_idx]
                    out_file['n_cells'][output_idx] = summary_stats['n_cells']
                    for k in ('sum', 'sumsq', 'gt0', 'gt1'):
                        out_file[k][output_idx, :] = summary_stats[k]


