import h5py
import json
import multiprocessing
import numpy as np
import os
import pathlib
import tempfile

from hierarchical_mapping.utils.utils import (
    _clean_up)

from hierarchical_mapping.utils.taxonomy_utils import (
    get_all_leaf_pairs)

from hierarchical_mapping.diff_exp.scores import (
    aggregate_stats,
    read_precomputed_stats)

from hierarchical_mapping.marker_selection.utils import (
    select_marker_genes)


def assign_types_to_chunk(
        query_gene_data,
        query_gene_names,
        taxonomy_tree,
        precompute_path,
        score_path,
        marker_genes_per_pair=30,
        n_processors=6,
        tmp_dir=None):
    """
    query_gene_data is an n_cells x n_genes chunk
    query_gene_names lists the names of the genes in this chunk

    A temporary HDF5 file with lists of marker genes for
    each parent node will be written out in tmp_path
    """

    tmp_dir = pathlib.Path(
        tempfile.mkdtemp(dir=tmp_dir))

    # precompute marker gene lists
    marker_gene_cache_path = tempfile.mkstemp(
            dir=tmp_dir,
            prefix='marker_lookup_',
            suffix='.h5')
    os.close(marker_gene_cache_path[0])
    marker_gene_cache_path = pathlib.Path(
        marker_gene_cache_path[1])

    mean_leaf_lookup = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precompute_path)

    _clean_up(tmp_dir)

def create_marker_gene_cache(
        cache_path,
        score_path,
        query_gene_names,
        taxonomy_tree,
        marker_genes_per_pair,
        n_processors=6):
    """
    Populate the temporary HDF5 file with the lists of marker
    genes for each parent node.

    Parameters
    ----------
    cache_path:
        The file to be written
    score_path:
        Path to the precomputed scores
    query_gene_names:
        list of gene names in the query dataset
    taxonomy_tree:
        Dict encoding the cell type taxonomy
    marker_genes_per_pair:
        Ideal number of marker genes to choose per leaf pair
    """

    parent_node_list = [None]
    hierarchy = taxonomy_tree['hierarchy']
    with h5py.File(cache_path, 'w') as out_file:
        out_file.create_group('None')
        for level in hierarchy[:-1]:
            out_file.create_group(level)
            these_parents = list(taxonomy_tree[level].keys())
            these_parents.sort()
            for parent in these_parents:
                parent_node_list.append((level, parent))

    for parent_node in parent_node_list:

        leaf_pair_list = get_all_leaf_pairs(
            taxonomy_tree=taxonomy_tree,
            parent_node=parent_node)

        if len(leaf_pair_list) > 0:
            marker_genes = select_marker_genes(
                score_path=score_path,
                leaf_pair_list=leaf_pair_list,
                query_genes=query_gene_names,
                genes_per_pair=marker_genes_per_pair,
                n_processors=n_processors,
                rows_at_a_time=1000000)
        else:
            marker_genes = {'reference': np.zeros(0, dtype=int),
                            'query': np.zeros(0, dtype=int)}

        with h5py.File(cache_path, 'a') as out_file:
            if parent_node is None:
                out_group = out_file['None']
            else:
                out_file[parent_node[0]].create_group(parent_node[1])
                out_group = out_file[parent_node[0]][parent_node[1]]
            out_group.create_dataset(
                'reference', data=marker_genes['reference'])
            out_group.create_dataset(
                'query', data=marker_genes['query'])


def get_leaf_means(
        taxonomy_tree,
        precompute_path):
    """
    Returns a lookup from leaf node name to mean
    gene expression array
    """
    precomputed_stats = read_precomputed_stats(precomputed_path)
    hierarchy = taxonomy_tree['hierarchy']
    leaf_level = hierarchy[-1]
    leaf_names = list(taxonomy_tree[leaf_level].keys())
    result = dict()
    for leaf in leaf_names:

        # gt1/0 threshold do not actually matter here
        stats = aggregate_stats(
                    leaf_population=[leaf,],
                    precomputed_stats=precomputed_stats,
                    gt0_threshold=1,
                    gt1_threshodl=0)
        result[leaf] = stats['mean']
    return result
