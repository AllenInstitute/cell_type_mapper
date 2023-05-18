import h5py
import json


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
