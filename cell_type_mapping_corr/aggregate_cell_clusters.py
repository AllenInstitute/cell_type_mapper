"""
This script takes an .h5ad file of cell-by-gene expression data and
produces an .h5 file containing a cluster-by-gene array representing
the average gene expression in each cell type cluster represented in
the original file.
"""

from typing import Union, Optional
import pathlib
import anndata
import argparse
import json
import numpy as np
import time
import h5py


def aggregate_clusters(
        anndata_path: Union[str, pathlib.Path],
        output_path: Union[str, pathlib.Path],
        gb_per_chunk: int = 19,
        n_cells: Optional[int] = None) -> None:
    """
    Average the cell-by-gene data over clusters and write out the
    cluster-by-gene file.

    Parameters
    ----------
    anndata_path:
        Path to the .h5ad file containing the cell-by-gene data

    output_path:
        Path to the .h5 file that will be written

    gb_per_chunk:
        Approximate GB to read in from anndata_path at a time.

    n_cells:
        For testing purposes. Only use the first n-cells cells.
        If None, use all cells.

    Returns
    -------
    None
        Data is written to output_path

    Notes
    -----
    output_path will contain 3 datasets. 'cluster_by_gene' is a numpy
    array of gene expression data. 'cluster_names' is the JSON-serialized
    list of cell cluster names. 'gene_names' is the JSON-serialized
    list of gene names.
    """
    data_src = anndata.read_h5ad(anndata_path, backed='r')

    metadata = get_metadata(data_src=data_src)

    if n_cells is not None:
        metadata['n_cells'] = n_cells

    rows_per_chunk = np.ceil(
            gb_per_chunk*1024**3/(8*metadata['n_genes'])).astype(int)

    print(f"{rows_per_chunk} rows per chunk")

    # create the arrays where the final result will be
    # aggregated
    final_ct = np.zeros(metadata['n_clusters'], dtype=int)

    # This is the transpose of the actual final output so that we
    # can do element-wise division at the end.
    final_sum = np.zeros((metadata['n_genes'],
                          metadata['n_clusters']),
                         dtype=float)

    cluster_to_row = metadata['cluster_to_row']
    t0 = time.time()
    chunk_ct = 0

    ntot = data_src.shape[0]//rows_per_chunk

    all_cluster_rows = np.array([cluster_to_row[c]
                                 for c in data_src.obs.cluster_label.values])

    data_iterator = data_src.chunked_X(rows_per_chunk)

    tot_cells = 0
    if n_cells is not None:
       ntot = n_cells // rows_per_chunk
    for chunk in data_iterator:
        cluster_rows = all_cluster_rows[chunk[1]:chunk[2]]
        data = chunk[0].toarray()
        this_chunk = aggregate_chunk(
                data=data,
                clusters=cluster_rows)
        for cluster in this_chunk['ct']:
            final_ct[cluster] += this_chunk['ct'][cluster]
            final_sum[:, cluster] += this_chunk['sum'][cluster]
        chunk_ct += 1
        duration = (time.time()-t0)/60.0
        per = duration/chunk_ct
        pred = ntot*per
        remain = pred-duration
        print(f"{chunk_ct} in {duration:.2e} minutes; "
              f"{remain:.2e} of {pred:.2e} left")

        if n_cells is not None:
            tot_cells += (chunk[2]-chunk[1])
            if tot_cells >= n_cells:
                break
    final_mean = (final_sum/final_ct).transpose()
    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset(
            'cluster_by_gene',
            data=final_mean,
            chunks=(min(1000, final_mean.shape[0]),
                    min(1000, final_mean.shape[1])))

        print("wrote cluster_by_gene")

        cluster_to_row = metadata['cluster_to_row']
        row_to_cluster = {cluster_to_row[c]:c
                          for c in cluster_to_row}
        cluster_names = [row_to_cluster[ii]
                         for ii in range(len(row_to_cluster))]
        out_file.create_dataset(
            'cluster_names',
            data=json.dumps(cluster_names).encode('utf-8'))

        out_file.create_dataset(
            'gene_names',
            data=json.dumps(metadata['gene_names']).encode('utf-8'))
 
    print("done")


def aggregate_chunk(
        data: np.ndarray,
        clusters: np.ndarray) -> dict:
    """
    Collect a chunk of expression data by cell cluster.

    Parameters
    ----------
    data:
        A cell-by-gene array of gene expression data

    clusters:
        A numpy.array of ints indicating which clusters the
        rows of data belong to

    Returns
    -------
    {'sum': a dict mapping cluster to the sum of the rows of data
            in that cluster
     'ct': a dict mapping cluster to the number of cells in that cluster}
    """

    sum_buffer = dict()
    ct_buffer = dict()

    unq_clusters = np.unique(clusters)
    for this_cluster in unq_clusters:
        valid = (clusters==this_cluster)
        sum_buffer[this_cluster] = data[valid, :].sum(axis=0)
        ct_buffer[this_cluster] = valid.sum()

    return {'sum': sum_buffer,
            'ct': ct_buffer}


def get_metadata(
        data_src: anndata.AnnData) -> dict:
    """
    Get a dict of the metadata needed to run the aggreagation.

    Parameters
    ----------
    data_src:
        The AnnData object representing the cell-by-gene data.

    Returns
    -------
    {'n_genes': the number of genes in data_src
     'n_cells': the number of cells in data_src
     'n_clusters': the number of cell clusters in data_src
     'cluster_to_row': a dict mapping cluster name to row index
                       in the final cluster-by-gene array
     'gene_names': the list of gene names in data_src}
    """

    n_genes = data_src.shape[1]
    n_cells = data_src.shape[0]
    cluster_names = set(data_src.obs.cluster_label.values)
    cluster_names = list(cluster_names)

    n_clusters = len(cluster_names)

    # the cluster names start with an integer prefix; we'll
    # sort by that
    idx_to_sort_by = [int(c.split()[0].split('_')[0])
                      for c in cluster_names]

    sorted_dex = np.argsort(idx_to_sort_by)

    cluster_to_row = dict()
    for i_row, i_cluster in enumerate(sorted_dex):
        cluster_to_row[cluster_names[i_cluster]] = i_row

    gene_names = list(data_src.var_names)

    return {'n_genes': n_genes,
            'n_cells': n_cells,
            'n_clusters': n_clusters,
            'cluster_to_row': cluster_to_row,
            'gene_names': gene_names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anndata_path', type=str, default=None,
                        help='path to the cell-by-gene .h5ad file')
    parser.add_argument('--output_path', type=str, default=None,
                        help=('path to the cluster-by-gene .h5 '
                              'file to be written'))
    parser.add_argument('--n_cells', type=int, default=None,
                        help=('if not None, only aggregate this many '
                              'cells (for testing)'))
    parser.add_argument('--gb_per_chunk', type=int, default=19,
                        help=('approximate GB to load from anndata '
                              'at a time'))
    args = parser.parse_args()

    aggregate_clusters(
        anndata_path=args.anndata_path,
        output_path=args.output_path,
        n_cells=args.n_cells,
        gb_per_chunk=args.gb_per_chunk)


if __name__ == "__main__":
    main()
