import h5py
import json
import numpy as np
import multiprocessing

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_list,
    DummyLock)

from hierarchical_mapping.corr.utils import (
    match_genes)

from hierarchical_mapping.binary_array.backed_binary_array import (
    BackedBinarizedBooleanArray)


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


def create_usefulness_array(
        cache_path,
        gb_size=10):
    """
    Create an (n_genes,) array of how useful each gene is as a marker.
    Usefulness is just a count of how many (+/-, taxonomy_pair) combinations
    the gene is a marker for (in this case +/- indicates which node in the
    taxonomy pair the gene is up-regulated for).

    Parameters
    ----------
    cache_path:
        path to the file created by markers.find_markers_for_all_taxonomy_pairs
    gb_size:
        Number of gigabytes to load at a time (approximately)

    Returns
    -------
    A numpy array of ints indicating the usefulness of each gene.

    Notes
    -----
    As implemented, it is assumed that the rows of the arrays in cache_path
    are genes and the columns are taxonomy pairs
    """

    with h5py.File(cache_path, "r", swmr=True) as src:
        n_cols = src['n_pairs'][()]
        n_rows = len(json.loads(src['gene_names'][()].decode('utf-8')))

    is_marker = BackedBinarizedBooleanArray(
        h5_path=cache_path,
        h5_group='markers',
        n_rows=n_rows,
        n_cols=n_cols,
        read_only=True)

    usefulness_sum = np.zeros(is_marker.n_rows, dtype=int)

    byte_size = gb_size*1024**3
    batch_size = max(1, np.round(byte_size/n_cols).astype(int))

    for row0 in range(0, n_rows, batch_size):
        row1 = min(n_rows, row0+batch_size)
        row_batch = is_marker.get_row_batch(row0, row1)
        usefulness_sum[row0:row1] = row_batch.sum(axis=1)

    return usefulness_sum
