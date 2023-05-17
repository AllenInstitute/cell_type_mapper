from typing import Union, Dict, List, Any, Optional
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

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list)

from hierarchical_mapping.utils.sparse_utils import (
    _load_disjoint_csr)

from hierarchical_mapping.utils.stats_utils import (
    summary_stats_for_chunk)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


def _create_empty_stats_file(
        output_path,
        cluster_to_output_row,
        n_clusters,
        n_genes,
        col_names=None):

    with h5py.File(output_path, 'w') as out_file:

        if col_names is not None:
            out_file.create_dataset(
                'col_names',
                data=json.dumps(col_names).encode('utf-8'))

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
