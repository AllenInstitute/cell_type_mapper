from typing import Union
import pathlib
import anndata
import argparse
import json
import numpy as np
import time
import h5py


def aggregate_clusters(
        anndata_path,
        output_path,
        n_cells=None):

    metadata = get_metadata(anndata_path=anndata_path)

    if n_cells is not None:
        metadata['n_cells'] = n_cells

    rows_per_chunk = np.ceil(
            19*1024**3/(8*metadata['n_genes'])).astype(int)

    print(f"{rows_per_chunk} rows per chunk")

    final_ct = np.zeros(metadata['n_clusters'], dtype=int)
    final_sum = np.zeros((metadata['n_genes'],
                          metadata['n_clusters']),
                         dtype=float)

    cluster_to_row = metadata['cluster_to_row']
    t0 = time.time()
    chunk_ct = 0
    data_src = anndata.read_h5ad(anndata_path, backed='r')
    ntot = data_src.shape[0]//rows_per_chunk
    all_cluster_rows = np.array([cluster_to_row[c] for c in data_src.obs.cluster_label.values])
    data_iterator = data_src.chunked_X(rows_per_chunk)
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
        data,
        clusters):

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
        anndata_path: Union[str, pathlib.Path]) -> dict:
    """
    Get the parameters needed to create the temporary HDF5 file
    for aggregating the cluster gene expression data.

    Parameters
    ----------
    anndata_path: Union[str, pathlib.Path]
        Path to the AnnData file whose rows are being aggregated

    Returns
    -------
    A dict. 'n_genes' will be the number of genes in the expression
    matrix. 'n_clusters' will be the number of clusters in the dataset.
    'cluster_to_row' will be a dict mapping cluster names
    to row numbers in the temporary HDF5 file to be produced.
    """

    data_src = anndata.read_h5ad(anndata_path, backed='r')
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
    parser.add_argument('--anndata_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--n_cells', type=int, default=None)
    args = parser.parse_args()

    aggregate_clusters(
        anndata_path=args.anndata_path,
        output_path=args.output_path,
        n_cells=args.n_cells)


if __name__ == "__main__":
    main()
