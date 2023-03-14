import h5py
import itertools
import json
import numpy as np
import multiprocessing
import time

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list,
    DummyLock)

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves)

from hierarchical_mapping.corr.utils import (
    match_genes)


def select_marker_genes(
        taxonomy_tree,
        parent_node,
        score_path,
        query_genes,
        genes_per_pair=30,
        n_processors=4,
        rows_at_a_time=100000):

    """
    Parameters
    ----------
    taxonomy_tree:
        A dict encoding the cell type taxonomy we are using
    parent_node:
        A tuple of the type (level, node) denoting the node
        we know these query cells belong to (so, we are selecting
        the marker genes for discribinating the level below this)

        If parent_node is None, then assume that we are selecting
        marker genes for the highest level of the taxonomy

    score_path:
        Path to the HDF5 file containing the ranked_list of
        marker genes for all taxonomic pairs in the
        reference dataset
    query_genes:
        List of gene names available in the query set.
    genes_per_pair:
        Number of marker genes to consider per leaf pair
    n_processors:
        The number of independent worker processes to spin up
    rows_at_a_time:
        Number of rows to load from score_path['ranked_list'] at
        a time (to avoid overrunning memory)

    Returns
    -------
    A dict.
        "reference" -> the array of indices of marker genes in the
        reference dataset

        "query" -> the array of indices of marker genes in the
        query dataset
    """

    hierarchy = taxonomy_tree['hierarchy']
    leaf_level = hierarchy[-1]

    if parent_node is not None:
        if parent_node[0] == leaf_level:
            raise RuntimeError(
                "No need to select marker genes; you are already "
                "in the leaf level of the taxonomy\n"
               f"parent_node: {parent_node}")

        # find the level in the hierarchy that is the immediate
        # child of parent_node[0]
        for child_level_idx, level in enumerate(hierarchy):
            if level == parent_node[0]:
                break
        child_level_idx += 1

        if child_level_idx > len(hierarchy):
            raise RuntimeError(
                f"Somehow, child_level_idx={child_level_idx}\n"
                f"while the hierarchy has {len(hierarchy)} levels;\n"
                f"parent_node = {parent_node}")
        child_level = hierarchy[child_level_idx]

        # all of the siblings that directly inherit from
        # parent_node[0]
        siblings = taxonomy_tree[parent_node[0]][parent_node[1]]
    else:
        siblings = list(taxonomy_tree[hierarchy[0]].keys())
        child_level = hierarchy[0]

    with h5py.File(score_path, 'r') as in_file:
        pair_to_idx = json.loads(
                in_file['pair_to_idx'][()].decode('utf-8'))
        reference_gene_names = json.loads(
                in_file['gene_names'][()].decode('utf-8'))

    # find the rows of score_path['ranked_list'] that we need to select
    # markers from by comparing, cluster-to-cluster, all of the
    # sibling pairs whose direct parent is parent_node[0]
    leaf_pair_idx_set = set()
    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)
    for sibling_pair in itertools.combinations(siblings, 2):
        leaf_list_0 = tree_as_leaves[child_level][sibling_pair[0]]
        leaf_list_1 = tree_as_leaves[child_level][sibling_pair[1]]
        for leaf_pair in itertools.product(leaf_list_0, leaf_list_1):
            idx = pair_to_idx[leaf_level][leaf_pair[0]][leaf_pair[1]]
            leaf_pair_idx_set.add(idx)

    leaf_pair_idx_arr = np.sort(np.array(list(leaf_pair_idx_set)))

    gene_overlap = match_genes(
            reference_gene_names=reference_gene_names,
            query_gene_names=query_genes)

    valid_reference_genes = set(gene_overlap['reference'])
    row0 = leaf_pair_idx_arr.min()
    marker_set = set()

    if n_processors > 1:
        mgr = multiprocessing.Manager()
        output_dict = mgr.dict()
        output_lock = mgr.Lock()
    else:
        mgr = None
        output_dict = dict()
        output_lock = DummyLock()

    t0 = time.time()
    process_list = []
    with h5py.File(score_path, 'r') as in_file:
        arr_shape = in_file['ranked_list'].shape
        n_chunks = arr_shape[0]
        while True:
            row1 = min(row0 + rows_at_a_time, arr_shape[0])
            rank_chunk = in_file['ranked_list'][row0:row1, :]

            if mgr is None:
                _process_rank_chunk_worker(
                    valid_rows=leaf_pair_idx_set,
                    valid_genes=valid_reference_genes,
                    rank_chunk=rank_chunk,
                    row0=row0,
                    row1=row1,
                    genes_per_pair=genes_per_pair,
                    output_dict=output_dict,
                    output_lock=output_lock)
                k_list = list(output_dict.keys())
                for k in k_list:
                    marker_set = marker_set.union(
                                    output_dict.pop(k))
            else:
                p = multiprocessing.Process(
                        target=_process_rank_chunk_worker,
                        kwargs={
                            'valid_rows': leaf_pair_idx_set,
                            'valid_genes': valid_reference_genes,
                            'rank_chunk': rank_chunk,
                            'row0': row0,
                            'row1': row1,
                            'genes_per_pair': genes_per_pair,
                            'output_dict': output_dict,
                            'output_lock': output_lock})
                p.start()
                process_list.append(p)
                while len(process_list) >= n_processors:
                    process_list = winnow_process_list(process_list)
                    if len(output_dict) > 0:
                        with output_lock:
                            k_list = list(output_dict.keys())
                            for k in k_list:
                                marker_set = marker_set.union(
                                                output_dict.pop(k))

            next_rows = np.where(leaf_pair_idx_arr >= row1)[0]
            if len(next_rows) == 0:
                break
            row0 = leaf_pair_idx_arr[next_rows.min()]
            print_timing(
                t0=t0,
                tot_chunks=n_chunks,
                i_chunk=row0,
                unit='hr')

    for p in process_list:
        p.join()
    k_list = list(output_dict.keys())
    for k in k_list:
        marker_set = marker_set.union(
                        output_dict.pop(k))

    return marker_set


def _process_rank_chunk_worker(
        valid_rows,
        valid_genes,
        rank_chunk,
        row0,
        row1,
        genes_per_pair,
        output_dict,
        output_lock):

    marker_set = _process_rank_chunk(
        valid_rows=valid_rows,
        valid_genes=valid_genes,
        rank_chunk=rank_chunk,
        row0=row0,
        row1=row1,
        genes_per_pair=genes_per_pair)

    with output_lock:
        output_dict[(row0, row1)] = marker_set


def _process_rank_chunk(
        valid_rows,
        valid_genes,
        rank_chunk,
        row0,
        row1,
        genes_per_pair):
    """
    Parameters
    ----------
    valid_rows:
        rows from the score file that are actually being considered
        (a set of integers)
    valid_genes:
        set of ints indicating genes that overlap between the reference
        and query sets
    rank_chunk:
        contiguous chunk of score_file['ranked_list'] data
    row0:
        The min row in rank_chunk
    row1:
        The max row in rank_chunk
    genes_per_pair:
        desired number of marker genes to select from each row

    Returns
    -------
    Set of integers denoting marker genes
    """
    row_mask = np.zeros(rank_chunk.shape[0], dtype=bool)
    for ii, row in enumerate(range(row0, row1, 1)):
        if row in valid_rows:
            row_mask[ii] = True
    rank_chunk = rank_chunk[row_mask, :]
    gene_mask = np.zeros(rank_chunk.shape[1], dtype=bool)

    marker_set = set()
    for i_row in range(rank_chunk.shape[0]):
        this_row = rank_chunk[i_row, :]
        gene_mask[:] = False
        for i_g, g in enumerate(this_row):
            if g in valid_genes:
                gene_mask[i_g] = True
        this_row = this_row[gene_mask]
        if len(this_row) > 0:
            marker_set = marker_set.union(set(this_row[:genes_per_pair]))
    return marker_set
