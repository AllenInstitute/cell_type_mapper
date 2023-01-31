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
        cell_path,
        cluster_path,
        output_path,
        cells_at_a_time=1000,
        tmp_dir=None,
        n_processors=6):

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

    for p in process_list:
        p.join()
    print("started all workers")


def _worker(
        cell_data_path,
        cluster_by_gene,
        cell_row_specification,
        output_lock,
        output_path,
        rows_at_a_time=5000):
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


def get_metadata(cell_path, cluster_path, tmp_dir):
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
        data_src,
        tmp_path,
        col_idx,
        chunk_size=5000):

    print(f"transcrbing data to\n{tmp_path}")
    n_chunks = data_src.shape[0]//chunk_size

    with h5py.File(tmp_path, 'w') as out_file:
        out_file.create_dataset(
            'data',
            shape=(data_src.shape[0], len(col_idx)),
            dtype=data_src.X.dtype,
            chunks=(min(1000, data_src.shape[0]),
                    min(1000, len(col_idx))))

        print("transcribing anndata")
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
        cell_by_gene,
        cluster_by_gene):
    """
    Compute the Pearson's correlation coefficient
    of every cell in cell_by_gene versus every cluster
    in cluster_by_gene
    """

    return 1.0-scipy_cdist(cell_by_gene,
                           cluster_by_gene,
                           metric='correlation')

def match_genes(
        anndata_src,
        cluster_path):
    """
    Returns a dict containing the names of genes that matched
    across the datasets as well as two lists containing the
    column index of those genes in the anndata dataset and
    the cluster-by-gene dataset
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

