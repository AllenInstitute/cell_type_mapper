import h5py
import json
import numpy as np
import multiprocessing
import time

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list,
    DummyLock)

from hierarchical_mapping.corr.utils import (
    match_genes)


def select_marker_genes(
        score_path,
        leaf_pair_list,
        query_genes,
        genes_per_pair=30,
        n_processors=4,
        rows_at_a_time=100000):

    """
    Parameters
    ----------
    score_path:
        Path to the HDF5 file containing the ranked_list of
        marker genes for all taxonomic pairs in the
        reference dataset
    leaf_pair_list:
        List of (level, node1, node2) tuples indicating the
        pairs of leaf nodes that need to be compared
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
    with h5py.File(score_path, 'r', swmr=True) as in_file:
        pair_to_idx = json.loads(
                in_file['pair_to_idx'][()].decode('utf-8'))
        reference_gene_names = json.loads(
                in_file['gene_names'][()].decode('utf-8'))

    # find the rows of score_path['ranked_list'] that we need to select
    # markers from by comparing, cluster-to-cluster, all of the
    # sibling pairs whose direct parent is parent_node[0]
    leaf_pair_idx_set = set()
    for leaf_pair in leaf_pair_list:
        idx = pair_to_idx[leaf_pair[0]][leaf_pair[1]][leaf_pair[2]]
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

    process_list = []
    with h5py.File(score_path, 'r', swmr=True) as in_file:
        arr_shape = in_file['ranked_list'].shape
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

    for p in process_list:
        p.join()
    k_list = list(output_dict.keys())
    for k in k_list:
        marker_set = marker_set.union(
                        output_dict.pop(k))

    marker_set = list(marker_set)
    marker_set.sort()
    result = dict()
    result['reference'] = marker_set
    ref_to_query = {rr: qq
                    for rr, qq in zip(gene_overlap['reference'],
                                      gene_overlap['query'])}
    result['query'] = [ref_to_query[rr] for rr in marker_set]
    return result


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


def create_utility_array(
        marker_gene_array,
        gb_size=10,
        taxonomy_mask=None):
    """
    Create an (n_genes,) array of how useful each gene is as a marker.
    Utility is just a count of how many (+/-, taxonomy_pair) combinations
    the gene is a marker for (in this case +/- indicates which node in the
    taxonomy pair the gene is up-regulated for).

    Parameters
    ----------
    marker_gene_array:
        A MarkerGeneArray
    gb_size:
        Number of gigabytes to load at a time (approximately)
    taxonomy_mask:
        if not None, a list of integers denoting which columns to
        sum utility for.

    Returns
    -------
    utility_arry:
        A numpy array of floats indicating the utility of each gene.

    marker_census:
        A numpy of ints indicating how many markers there are for
        each (taxonomy pair, sign) combination.

    Notes
    -----
    As implemented, it is assumed that the rows of the arrays in cache_path
    are genes and the columns are taxonomy pairs
    """

    is_marker = marker_gene_array.is_marker
    up_regulated = marker_gene_array.up_regulated
    n_cols = marker_gene_array.n_pairs
    n_rows = marker_gene_array.n_genes

    if taxonomy_mask is None:
        n_taxon = n_cols
    else:
        n_taxon = len(taxonomy_mask)
    marker_census = np.zeros((n_taxon, 2), dtype=int)
    utility_sum = np.zeros(is_marker.n_rows, dtype=int)

    byte_size = gb_size*1024**3
    batch_size = max(1, np.round(byte_size/(3*n_cols)).astype(int))

    t0 = time.time()
    for row0 in range(0, n_rows, batch_size):
        row1 = min(n_rows, row0+batch_size)
        up_reg_batch = up_regulated.get_row_batch(row0, row1)
        marker_batch = is_marker.get_row_batch(row0, row1)

        if taxonomy_mask is not None:
            marker_batch = marker_batch[:, taxonomy_mask]
            up_reg_batch = up_reg_batch[:, taxonomy_mask]

        utility_sum[row0:row1] = marker_batch.sum(axis=1)
        marker_census[:, 0] += (np.logical_not(up_reg_batch)
                                * marker_batch).sum(axis=0)
        marker_census[:, 1] += (up_reg_batch*marker_batch).sum(axis=0)
    duration = (time.time()-t0)
    print(f"got census in {duration:.2e} seconds")

    return utility_sum, marker_census
