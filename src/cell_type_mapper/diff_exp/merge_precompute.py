"""
Define function to merge precomputed stats files
"""

import h5py
import json
import numpy as np
import pathlib

from cell_type_mapper.taxonomy.merger import merge_taxonomy_trees
from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree


def merge_precomputed_stats_files(
        src_lookup,
        dst_path,
        clobber=False):
    """
    Parameters
    ----------
    src_lookup:
        dict mapping taxonomy name to the path to
        the precomputed_stats file for that taxonomy
    dst_path:
        path to the file being written
    clobber:
        if False and dst_path exists, fail. If True,
        overwrite
    """
    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not dst_path.is_file():
            raise RuntimeError(
                f"{dst_path} exists, but is not a file"
            )
        if not clobber:
            raise RuntimeError(
                f"{dst_path} exists; run with clobber=True "
                "to overwrite"
            )
        else:
            dst_path.unlink()

    tree_lookup = {
        name: TaxonomyTree.from_precomputed_stats(src_lookup[name])
        for name in src_lookup
    }

    merged_tree, name_map = merge_taxonomy_trees(tree_lookup)
    output_cluster_to_row = {
        cl: ii
        for ii, cl in enumerate(
            merged_tree.nodes_at_level(merged_tree.leaf_level)
        )
    }

    n_clusters = len(output_cluster_to_row)

    gene_idx_lookup = dict()
    raw_row_idx_lookup = dict()
    key_lookup = dict()
    dtype_lookup = None

    for taxonomy_name in src_lookup:
        with h5py.File(src_lookup[taxonomy_name], 'r') as src:
            gene_idx_lookup[taxonomy_name] = {
                g: ii
                for ii, g in enumerate(
                    json.loads(src['col_names'][()].decode('utf-8'))
                )
            }
            raw_row_idx_lookup[taxonomy_name] = json.loads(
                src['cluster_to_row'][()].decode('utf-8')
            )
            key_lookup[taxonomy_name] = set([
                k for k in src.keys()
                if np.issubdtype(src[k], np.number)
            ])

            if dtype_lookup is None:
                dtype_lookup = {
                    k: src[k].dtype for k in src.keys()
                }

    # create a dict mapping taxonomy name to a numpy array of
    # integers for input array and a numpy array of integers
    # for output array
    row_idx_lookup = dict()
    for taxonomy_name in raw_row_idx_lookup:
        leaf_level = tree_lookup[taxonomy_name].leaf_level
        cl_names = sorted(raw_row_idx_lookup[taxonomy_name].keys())
        input_idx = np.array(
            [raw_row_idx_lookup[taxonomy_name][cl] for cl in cl_names]
        )
        output_idx = np.array(
            [output_cluster_to_row[
                    name_map[taxonomy_name][leaf_level][cl]['node']
             ]
             for cl in cl_names]
        )
        sorted_dex = np.argsort(output_idx)
        output_idx = output_idx[sorted_dex]
        input_idx = input_idx[sorted_dex]
        row_idx_lookup[taxonomy_name] = {
            'input': input_idx,
            'output': output_idx
        }

    final_gene_set = None
    for taxonomy_name in gene_idx_lookup:
        gene_set = set(gene_idx_lookup[taxonomy_name].keys())
        if final_gene_set is None:
            final_gene_set = gene_set
        else:
            final_gene_set = final_gene_set.intersection(gene_set)

    if len(final_gene_set) == 0:
        raise RuntimeError(
            "No overlap in genes contained within the "
            "given precomputed_stats files"
        )

    final_gene_set = sorted(final_gene_set)
    n_genes = len(final_gene_set)
    input_gene_idx_lookup = dict()
    for taxonomy_name in gene_idx_lookup:
        input_gene_idx_lookup[taxonomy_name] = np.array(
            [gene_idx_lookup[taxonomy_name][g] for g in final_gene_set]
        )

    numerical_key_set = None
    for taxonomy_name in key_lookup:
        if numerical_key_set is None:
            numerical_key_set = key_lookup[taxonomy_name]
        else:
            numerical_key_set = numerical_key_set.intersection(
                key_lookup[taxonomy_name]
            )

    if len(numerical_key_set) == 0:
        raise RuntimeError(
            "No overlap in numerical datasets between "
            "the gien precomputed_stats files"
        )

    with h5py.File(dst_path, 'w') as dst:
        dst.create_dataset(
            'taxonomy_tree',
            data=merged_tree.to_str(drop_cells=True).encode('utf-8')
        )
        dst.create_dataset(
            'col_names',
            data=json.dumps(final_gene_set).encode('utf-8')
        )
        dst.create_dataset(
            'cluster_to_row',
            data=json.dumps(output_cluster_to_row).encode('utf-8')
        )
        for key in numerical_key_set:
            if key == 'n_cells':
                dst.create_dataset(
                    'n_cells',
                    shape=n_clusters,
                    dtype=int
                )
            else:
                dst.create_dataset(
                    key,
                    shape=(n_clusters, n_genes),
                    dtype=dtype_lookup[key]
                )

        for taxonomy_name in src_lookup:
            input_rows = row_idx_lookup[taxonomy_name]['input']
            output_rows = row_idx_lookup[taxonomy_name]['output']
            input_genes = input_gene_idx_lookup[taxonomy_name]
            with h5py.File(src_lookup[taxonomy_name], 'r') as src:
                for key in numerical_key_set:
                    if key == 'n_cells':
                        dst[key][output_rows] = src[key][()][input_rows]
                    else:
                        dst[key][output_rows, :] = (
                            src[key][()][input_rows, :][:, input_genes]
                        )
