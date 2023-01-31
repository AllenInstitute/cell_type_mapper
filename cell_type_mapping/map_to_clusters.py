"""
Implementation of a very simple scheme to map cells of unknown
type onto a gene-expression-based taxonomy.

Mapping is based on the cross-correlation of each cell with
each cell cluster along the 'genes' axis.
"""

from typing import Union, Tuple

import anndata
import pathlib
import h5py
import json
import time
import argparse
import numpy as np
import tempfile
import multiprocessing

from scipy.spatial.distance import cdist as scipy_cdist


def map_cells_to_clusters(
        cell_path: Union[str, pathlib.Path],
        cluster_path: Union[str, pathlib.Path],
        output_path: Union[str, pathlib.Path],
        cells_at_a_time: int =1000,
        tmp_dir: Optional[Union[str, pathlib.Path]] = None,
        n_processors: int = 6) -> None:
    """
    Map the cells from a cell-by-gene expression array onto the clusters
    from a cluster-by-gene expression array using cross correlation

    Parameters
    ----------
    cell_path:
        Path to the .h5ad file containing the cell-by-gene data

    cluster_path:
        Path to the .h5 file (as produced by aggregate_cell_clusters.py)
        containing the cluster-by-gene data.

    output_path:
        Path to the .h5 file that will be written containing the
        cell-by-cluster correlation data

    cells_at_a_time:
        Number of cells to load in one "chunk"

    tmp_dir:
       Directory where a temporary HDF5 file is written containing
       the cell-by-gene data (Running this script on the MERFISH data,
       the array is small enough that it makes sense to write it out
       to a chunked HDF5 file for more rapid access)

    n_processors:
        The number of independent workers to spin up with
        multiprocessing

    Returns
    -------
    None
        Results are written to the file specified by output_path
    """

    gene_lookup = get_metadata(
                cell_path=cell_path,
                cluster_path=cluster_path,
                tmp_dir=tmp_dir)

    with h5py.File(cluster_path, 'r') as in_file:
        cluster_by_gene = in_file['cluster_by_gene'][()]
        cluster_names = in_file['cluster_names'][()]

    cluster_by_gene = cluster_by_gene[:, gene_lookup['cluster_idx']]

    n_clusters = cluster_by_gene.shape[0]
    n_cells = gene_lookup['n_cells']

    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset(
            'cell_by_cluster',
            shape=(n_cells, n_clusters),
            dtype=float,
            chunks=(min(n_cells, 1000),
                    min(n_clusters, 1000)))

        out_file.create_dataset(
            'cluster_names',
            data=cluster_names)

        out_file.create_dataset(
            'cell_identifiers',
            data=json.dumps(list(gene_lookup['cell_names'])).encode('utf-8'))

        out_file.create_dataset(
            'gene_names',
            data=json.dumps(gene_lookup['gene_names']).encode('utf-8'))

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    process_list = []
    row_spec_list = []
    cells_per_process = np.ceil(n_cells/n_processors).astype(int)
    for i0 in range(0, n_cells, cells_per_process):
        i1 = min(n_cells, i0+cells_per_process)
        row_spec_list.append((i0, i1))

    print("correlating cells with clusters")
    for row_spec in row_spec_list:
        p = multiprocessing.Process(
                target=_worker,
                kwargs={'cluster_by_gene': cluster_by_gene,
                        'cell_data_path': gene_lookup['tmp_cell_path'],
                        'cell_row_specification': row_spec,
                        'output_lock': output_lock,
                        'output_path': output_path,
                        'rows_at_a_time': 5000})
        p.start()
        process_list.append(p)

    print("started all workers")
    for p in process_list:
        p.join()
    print("done")

    tmp_path = pathlib.Path(gene_lookup['tmp_cell_path'])
    if tmp_path.exists():
        print(f"cleaning {tmp_path}")
        tmp_path.unlink()


def _worker(
        cell_data_path: Union[str, pathlib.Path],
        cluster_by_gene: np.ndarray,
        cell_row_specification: Tuple[int, int],
        output_lock: Any,
        output_path: Union[str, pathlib.Path],
        rows_at_a_time: int = 5000) -> None:
    """
    worker function to calculate the correlation coefficients
    for one chunk of the final cell-by-gene array and write
    them to the output file.

    Parameters
    ----------
    cell_data_path:
        Path to the h5 file (*not* the h5ad file) containing the
        cell-by-gene data

    cluster_by_gene:
        The array of cluster-by-gene expression data (already
        read in as a numpy array)

    cell_row_specification:
        (row_min, row_max) of the chunk being processed

    output_lock:
        A multiprocessing.Manager.Lock to prevent more than
        one thread from writing to the output file at once

    output_path:
        Path to the h5 file being written

    rows_at_a_time:
        The number of rows to process at once (this worker will
        load subsets of cell_row_specification at a time to avoid
        overwhelming memory)

    Returns
    -------
    None
        Results are written to output_path
    """
    t0 = time.time()
    row_min = cell_row_specification[0]
    row_max = cell_row_specification[1]
    nchunks = (row_max-row_min)//rows_at_a_time
    ct = 0
    for row0 in range(row_min, row_max, rows_at_a_time):
        row1 = min(row_max, row0+rows_at_a_time)
        with h5py.File(cell_data_path, 'r', swmr=True) as in_file:
            cell_by_gene = in_file['data'][row0:row1, :]

        corr = correlate_cells(
                    cell_by_gene=cell_by_gene,
                    cluster_by_gene=cluster_by_gene)

        with output_lock:
            with h5py.File(output_path, 'a') as out_file:
                out_file['cell_by_cluster'][row0:row1, :] = corr

        ct += 1
        if ct % 5 == 0:
            print_timing(t0=t0, i_chunk=ct, n_chunks=nchunks)


def get_metadata(
        cell_path: Union[str, pathlib.Path],
        cluster_path: Union[str, pathlib.Path],
        tmp_dir: Union[str, pathlib.Path]):
    """
    Perform metadata calculations needed to proceed with
    cell-to-cluster mapping.

    Parameters
    ----------
    cell_path:
        Path to the h5ad file containing the cell-by-gene data

    cluster_path:
        Path to the h5 file containing the cluster-by-gene data

    tmp_dir:
        Directory where a temporary HDF5 file containing the chunked
        cell-by-gene data is stored

    Returns
    -------
    {'gene_names': name of the genes matched in cell-by-gene and
                   cluster-by-gene data
     'cluster_idx': np.ndarray of column indices in cluster_by_gene
                    matching gene_names
     'annddata_idx': np.ndarray of column indicies in cell-by-gene data
                     matching gene_names
     'n_cells': number of cells in cell-by-gene data
     'cell_names': names of cells in cell-by-gene data (a list)
     'tmp_cell_path': temporary HDF5 file containing the
                      cell-by-gene data}

    Notes
    -----
    This function creates and populates tmp_cell_path
    """

    cell_src = anndata.read_h5ad(cell_path, backed='r')
    gene_lookup = match_genes(
                    anndata_src=cell_src,
                    cluster_path=cluster_path)

    n_cells = cell_src.shape[0]
    gene_lookup['n_cells'] = n_cells
    gene_lookup['cell_names'] = cell_src.obs_names

    print(f"matched {len(gene_lookup['gene_names'])} genes")

    tmp_path = pathlib.Path(
            tempfile.mkstemp(dir=tmp_dir,
                             suffix='.h5')[1])
    write_anndata_to_tmp(
        data_src=cell_src,
        tmp_path=tmp_path,
        col_idx=gene_lookup['anndata_idx'])

    gene_lookup['tmp_cell_path'] = tmp_path
    return gene_lookup


def write_anndata_to_tmp(
        data_src: anndata.AnnData,
        tmp_path: Union[str, pathlib.Path],
        col_idx: np.ndarray,
        chunk_size: int = 5000) -> None:
    """
    Write the cell-by-gene data as a dense array in a chunked
    HDF5 file (this is more efficient in the case of mapping our
    MERFISH data to the cell clusters in our 10X data)

    Parameters
    ----------
    data_src:
        The AnnData object containing the cell-by-gene data

    tmp_path:
        Path to the HDF5 file being written

    col_idx:
        np.ndarray of column indices corresponding to
        the genes being kept

    chunk_size:
        Number of rows read in from data_src at a time

    Returns
    -------
    None
        Results are written out to the file specified
        by tmp_path
    """

    print(f"transcrbing data to\n{tmp_path}")
    n_chunks = data_src.shape[0]//chunk_size

    with h5py.File(tmp_path, 'w') as out_file:
        out_file.create_dataset(
            'data',
            shape=(data_src.shape[0], len(col_idx)),
            dtype=data_src.X.dtype,
            chunks=(min(1000, data_src.shape[0]),
                    min(1000, len(col_idx))))

        t0 = time.time()
        for i_chunk, chunk in enumerate(data_src.chunked_X(chunk_size)):
            this = chunk[0].toarray()[:, col_idx]
            out_file['data'][chunk[1]:chunk[2], :] =  this
            if i_chunk % max(1, n_chunks//5) == 0:
                print_timing(
                    t0=t0,
                    i_chunk=i_chunk+1,
                    n_chunks=n_chunks)



def correlate_cells(
        cell_by_gene: np.ndarray,
        cluster_by_gene: np.ndarray):
    """
    Compute the Pearson's correlation coefficient
    of every cell in cell_by_gene versus every cluster
    in cluster_by_gene

    Parameters
    ----------
    cell_by_gene:
        (n_cells, n_genes)

    cluster_by_gene:
        (n_clusters, n_genes)

    Returns
    -------
    cell_by_cluster:
        (n_cells, n_clusters)
    """

    return 1.0-scipy_cdist(cell_by_gene,
                           cluster_by_gene,
                           metric='correlation')

def match_genes(
        anndata_src: anndata.AnnData,
        cluster_path: Union[str, pathlib.Path]) -> dict:
    """
    Returns a dict containing the names of genes that matched
    across the datasets as well as two lists containing the
    column index of those genes in the anndata dataset and
    the cluster-by-gene dataset

    Parameters
    ----------
    anndata_src:
        AnnData object containing cell-by-gene data

    cluster_path:
        Path to the h5 file containing the cluster-by-gene data

    Returns
    -------
    {'gene_names': list of names of genes that occur in both datasets
     'cluster_idx': np.ndarray of column indices needed to get those
                    genes from cluster-by-gene data
     'anndata_idx': np.ndarray of column indices needed to get those
                    genes from cell-by-gene data}
    """

    with h5py.File(cluster_path, 'r') as in_file:
        cluster_gene_names = json.loads(
                in_file['gene_names'][()].decode('utf-8'))

    anndata_gene_names = list(anndata_src.var_names)

    cluster_gene_set = set(cluster_gene_names)

    valid_gene_names = [g for g in anndata_gene_names
                        if g in cluster_gene_set]

    if len(valid_gene_names) == 0:
       raise RuntimeError("no gene name overlap")

    g_to_idx_cluster = {g:idx for idx, g in enumerate(cluster_gene_names)}
    g_to_idx_anndata = {g:idx for idx, g in enumerate(anndata_gene_names)}

    anndata_idx = [g_to_idx_anndata[g] for g in valid_gene_names]
    cluster_idx = [g_to_idx_cluster[g] for g in valid_gene_names]

    return {'gene_names': valid_gene_names,
            'cluster_idx': np.array(cluster_idx),
            'anndata_idx': np.array(anndata_idx)}

def print_timing(
        t0,
        i_chunk,
        n_chunks):
    """
    Utility function to print out a timing message to stdout
    """
    duration = (time.time()-t0)/60.0
    per = duration/i_chunk
    pred = n_chunks*per
    remain = pred-duration
    print(f"{i_chunk} of {n_chunks} in {duration:.2e} minutes; "
         f"{remain:.2e} of {pred:.2e} left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_data_path', type=str, default=None,
                        help=('Path to the h5ad file containing data for '
                              'the cells that need to be mapped'))
    parser.add_argument('--cluster_data_path', type=str, default=None,
                        help=('Path to the h5 file containing the cluster '
                              'by gene average expression matrix'))
    parser.add_argument('--output_path', type=str, default=None,
                        help=('Path to the h5 file that will be created'))
    parser.add_argument('--cells_at_a_time', type=int, default=1000,
                        help=('Number of cells to load at a time; '
                              'default=1000'))

    args = parser.parse_args()

    map_cells_to_clusters(
        cell_path=args.cell_data_path,
        cluster_path=args.cluster_data_path,
        output_path=args.output_path,
        cells_at_a_time=args.cells_at_a_time,
        tmp_dir='/local1/scott_daniel/scratch')

if __name__ == "__main__":
    main()

