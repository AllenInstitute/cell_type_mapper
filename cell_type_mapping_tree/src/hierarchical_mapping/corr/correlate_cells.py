import h5py
import json
import numpy as np
import time
from scipy.spatial.distance import cdist as scipy_cdist

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.corr.utils import (
    match_genes)

from hierarchical_mapping.utils.sparse_utils import (
    load_csr)


def correlate_cells(
        query_path,
        precomputed_path,
        output_path,
        gb_at_a_time=16):
    """
    query_path is the path to the h5ad file containing the query cells

    precomputed_path is the path to the precomputed stats file

    output_path is the path to the HDF5 file that will be written
    correlating cells with clusters
    """

    with h5py.File(query_path, 'r') as query_file:
        query_genes = [s.decode('utf-8')
                       for s in query_file['var/_index'][()]]

        query_indptr = query_file['X/indptr'][()]
        n_query_rows = len(query_indptr)-1
        n_query_cols = len(query_genes)
        rows_at_a_time = np.round(gb_at_a_time*1024**3/(8*n_query_cols)).astype(int)
        print(f"rows at a time {rows_at_a_time:.2e}")

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

            print("starting correlation")
            t0 = time.time()
            row_ct = 0
            with h5py.File(output_path, 'w') as out_file:
                for r0 in range(0, n_query_rows, rows_at_a_time):
                    r1 = min(n_query_rows, r0+rows_at_a_time)
                    print(r0, r1, rows_at_a_time)
                    query_chunk = load_csr(
                            row_spec=(r0, r1),
                            n_cols=n_query_cols,
                            data=query_file['X/data'],
                            indices=query_file['X/indices'],
                            indptr=query_indptr)

                    print("query_chunk",query_chunk.shape)

                    query_chunk = query_chunk[:, gene_idx['query']]

                    corr = 1.0-scipy_cdist(query_chunk,
                                           reference_profiles,
                                           metric='correlation')            

                    row_ct += query_chunk.shape[0]
                    print_timing(
                        t0=t0,
                        i_chunk=row_ct,
                        tot_chunks=n_query_rows,
                        unit='hr')
