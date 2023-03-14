import json
import h5py
import multiprocessing
import numpy as np
import os
import pathlib
import tempfile
import time

from hierarchical_mapping.utils.utils import (
    print_timing,
    file_size_in_bytes,
    _clean_up)

from hierarchical_mapping.utils.multiprocessing_utils import (
    DummyLock)

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves,
    get_all_pairs)

from hierarchical_mapping.utils.stats_utils import (
    welch_t_test,
    correct_ttest)


def score_all_taxonomy_pairs(
        precomputed_stats_path,
        taxonomy_tree,
        output_path,
        gt1_threshold=0,
        gt0_threshold=1,
        flush_every=1000,
        n_processors=4,
        tmp_dir=None,
        keep_all_stats=True,
        genes_to_keep=None):
    """
    Create differential expression scores and validity masks
    for differential genes between all relevant pairs in a
    taxonomy*

    * relevant pairs are defined as nodes in the tree that are
    on the same level and share the same parent.

    Parameters
    ----------
    precomputed_stats_path:
        Path to HDF5 file containing precomputed stats for leaf nodes

    taxonomy_tree:
        Dict encoding the taxonomy tree (created when we create the
        contiguous zarr file and stored in that file's metadata.json)

    output_path:
        Path to the HDF5 file where results will be stored

    gt1_thresdhold/gt0_threshold:
        Number of cells that must express above 0/1 in order to be
        considered a valid differential gene.

    flush_every:
        Write to HDF5 every flush_every pairs

    n_processors:
        Number of independent worker processes to spin out

    keep_all_stats:
        If True, keep scores and validity as well as ranked_list
        for each cell type pair. If False,  only keep ranked_list.
        (True results in a much larger file than False)

    genes_to_keep:
        If None, store data for all genes. If an integer, keep only the
        top-ranked genes_to_keep genes.

        Since validity, scores and ranked_list are not in the same order,
        can only call non-none genes_to_keep if keep_all_stats is False.

    Returns
    --------
    None
        Data is written to HDF5 file

    Notes
    -----
    HDF5 file will contain the following datasets
        'pair_to_idx' -> JSONized dict mapping [level][node1][node2] to row
        index in other data arrays

        'scores' -> (n_sibling_pairs, n_genes) array of differential expression
        scores

        'validity' -> (n_sibling_pairs, n_genes) array of booleans indicating
        whether or not the gene passed the validity thresholds

        'ranked_list' -> (n_sibling_pairs, n_genes) array of ints. Each row gives
        the ranked indexes of the discriminator genes, i.e. if
        ranked_list[2, :] = [9, 1,...., 101] then, for sibling pair at
        idx=2 (see pair_to_idx), gene_9 is the best discriminator, gene_1 is
        the second best discrminator, and gene_101 is the worst discriminator
    """

    if genes_to_keep is not None and keep_all_stats:
        raise RuntimeError(
            "Cannot have non-None genes_to_keep if "
            "keep_all_stats is True; "
            f"you have genes_to_keep={genes_to_keep}")

    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    tmp_dir = pathlib.Path(tmp_dir)

    hierarchy = taxonomy_tree['hierarchy']

    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)

    precomputed_stats = read_precomputed_stats(
           precomputed_stats_path)
    cluster_stats = precomputed_stats['cluster_stats']
    gene_names = precomputed_stats['gene_names']
    del precomputed_stats

    n_genes = len(gene_names)

    n_genes_to_keep = n_genes
    if genes_to_keep is not None:
        n_genes_to_keep = genes_to_keep

    idx_to_pair = _prep_output_file(
            output_path=output_path,
            taxonomy_tree=taxonomy_tree,
            n_genes=n_genes,
            n_genes_to_keep=n_genes_to_keep,
            keep_all_stats=keep_all_stats,
            gene_names=gene_names)

    print("starting to score")
    t0 = time.time()

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()

    process_list = []
    tmp_path_list = []
    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    n_per = len(idx_values)//n_processors
    for i_process in range(n_processors):
        i0 = i_process*n_per
        i1 = i0+n_per
        if i_process == n_processors-1:
            i1 = len(idx_values)

        tmp_path = tempfile.mkstemp(
                        dir=tmp_dir,
                        suffix='.h5')
        os.close(tmp_path[0])
        tmp_path = pathlib.Path(tmp_path[1])

        this_idx_values = idx_values[i0:i1]
        this_idx_to_pair = {
            ii: idx_to_pair.pop(ii)
            for ii in this_idx_values}

        (this_cluster_stats,
         this_tree_as_leaves) = _get_this_cluster_stats(
            cluster_stats=cluster_stats,
            idx_to_pair=this_idx_to_pair,
            tree_as_leaves=tree_as_leaves)

        p = multiprocessing.Process(
                target=_score_pairs_worker,
                kwargs={
                    'cluster_stats': this_cluster_stats,
                    'tree_as_leaves': this_tree_as_leaves,
                    'idx_to_pair': this_idx_to_pair,
                    'idx_values': this_idx_values,
                    'n_genes': n_genes,
                    'gt0_threshold': gt0_threshold,
                    'gt1_threshold': gt1_threshold,
                    'flush_every': flush_every,
                    'tmp_path': tmp_path,
                    'keep_all_stats': keep_all_stats,
                    'genes_to_keep': genes_to_keep,
                    'output_path': output_path,
                    'output_lock': output_lock})
        p.start()
        process_list.append(p)
        tmp_path_list.append(tmp_path)

    del cluster_stats
    del tree_as_leaves
    del this_cluster_stats
    del this_idx_values
    del this_idx_to_pair
    del this_tree_as_leaves

    for p in process_list:
        p.join()

    _clean_up(tmp_dir)
    duration = time.time()-t0
    print(f"that took {duration/3600.0:.2e} hrs")



def _get_this_cluster_stats(
        cluster_stats,
        idx_to_pair,
        tree_as_leaves):
    """
    Take global cluster_stats and an idx_to_pair dict.
    Return cluster_stats containing only those clusters that
    participate in idx_to_pair

    Also returns a sub-sampled tree_as_leaves

    Returns
    -------
    this_cluster_stats
    this_tree_as_leaves
    """
    t0 = time.time()
    leaf_set = set()
    this_cluster_stats = dict()
    this_tree_as_leaves = dict()
    for idx in idx_to_pair:
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        if level not in this_tree_as_leaves:
            this_tree_as_leaves[level] = dict()
        for node in (sibling_pair[1], sibling_pair[2]):
            if node not in this_tree_as_leaves[level]:
                this_tree_as_leaves[level][node] = tree_as_leaves[level][node]
                for leaf in tree_as_leaves[level][node]:
                    leaf_set.add(leaf)

    for node in leaf_set:
        this_cluster_stats[node] = cluster_stats[node]

    dur = time.time()-t0
    print(f"getting this_cluster_stats took {dur:.2e} seconds -- {len(leaf_set)} nodes")
    return this_cluster_stats, this_tree_as_leaves


def _prep_output_file(
       output_path,
       taxonomy_tree,
       n_genes,
       n_genes_to_keep,
       keep_all_stats,
       gene_names):
    """
    Create the HDF5 file where the differential gene scoring stats
    will be stored.

    Parameters
    ----------
    output_path:
        Path to the HDF5 file
    taxonomy_tree:
        Dict encoding the taxonomy tree (created when we create the
        contiguous zarr file and stored in that file's metadata.json)
    n_genes:
        Total number of genes in the data set
    n_genes_to_keep:
        Number of genes we are actually keeping
        (should only differ from n_genes if keep_all_stats
        is False)
    keep_all_stats:
        True if we are storing scores and validity;
        False otherwise
    gene_names:
        Ordered list of gene names for entire dataset

    Returns
    -------
    idx_to_pair:
        Dict mapping the row index of a sibling pair
        in the final output file to that sibling pair's
        (level, node1, node2) specification.

    Notes
    -----
    This method also creates the file at output_path with
    empty datasets for the stats that need to be saved.
    """
    siblings = get_all_pairs(taxonomy_tree)
    n_sibling_pairs = len(siblings)
    print(f"{n_sibling_pairs:.2e} sibling pairs")

    idx_to_pair = dict()
    pair_to_idx_out = dict()
    for idx, sibling_pair in enumerate(siblings):
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]
        idx_to_pair[idx] = sibling_pair

        if level not in pair_to_idx_out:
            pair_to_idx_out[level] = dict()
        if node1 not in pair_to_idx_out[level]:
            pair_to_idx_out[level][node1] = dict()
        if node2 not in pair_to_idx_out[level]:
            pair_to_idx_out[level][node2] = dict()

        pair_to_idx_out[level][node1][node2] = idx
        pair_to_idx_out[level][node2][node1] = idx

    chunk_size = (max(1, min(1000000//n_genes_to_keep, n_sibling_pairs)),
                  n_genes_to_keep)

    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset(
            'gene_names',
            data=json.dumps(gene_names).encode('utf-8'))

        out_file.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_out).encode('utf-8'))

        out_file.create_dataset(
            'ranked_list',
            shape=(n_sibling_pairs, n_genes_to_keep),
            dtype=int,
            chunks=chunk_size,
            compression='gzip')

        if keep_all_stats:
            out_file.create_dataset(
                'validity',
                shape=(n_sibling_pairs, n_genes),
                dtype=bool,
                chunks=chunk_size,
                compression='gzip')

            out_file.create_dataset(
                'scores',
                shape=(n_sibling_pairs, n_genes),
                dtype=float,
                chunks=chunk_size)

    return idx_to_pair

def _score_pairs_worker(
        cluster_stats,
        tree_as_leaves,
        idx_to_pair,
        idx_values,
        n_genes,
        gt0_threshold,
        gt1_threshold,
        flush_every,
        tmp_path,
        keep_all_stats,
        genes_to_keep,
        output_path,
        output_lock=None):
    """
    Score and rank differentiallly expressed genes for
    a subset of taxonomic siblings. Write the results to
    a temporary HDF5 file, then write the contents of that
    file to the final output file.

    Parameters
    ----------
    cluster_stats:
        Result of read_precomputed_stats (just 'cluster_stats')
    tree_as_leaves:
        Result of convert_tree_to_leaves
    idx_to_pair:
        Dict mapping row in output file to
        (level, node1, node2) sibling pair
    idx_values:
        Row indexes to be processed by this worker
    n_genes:
        Number of genes in dataset
    gt0_threshold:
        How many cells must express a gene above 0 for
        it to be valid
    gt1_threshold:
        How many cells must express a gene above 1 for
        it to be valid
    flush_every:
        Write to temporary output file every flush_every rows
    tmp_path:
        Path to temporary HDF5 file where results for this worker
        will be stored (this process creates and destroys that file)
    keep_all_stats:
        If True, keep scores and validity as well as ranked_list
        for each cell type pair. If False,  only keep ranked_list.
        (True results in a much larger file than False)
    genes_to_keep:
        If None, store data for all genes. If an integer, keep only the
        top-ranked genes_to_keep genes.

        Since validity, scores and ranked_list are not in the same order,
        can only call non-none genes_to_keep if keep_all_stats is False.
    output_path:
        Final output file
    output_lock:
        Multiprocessing lock to prevent multiple workers from writing
        to file at output_path simultaneously
    """
    if genes_to_keep is not None and keep_all_stats:
        raise RuntimeError(
            "Cannot have non-None genes_to_keep if "
            "keep_all_stats is True; "
            f"you have genes_to_keep={genes_to_keep}")

    n_genes_to_keep = n_genes
    if genes_to_keep is not None:
        n_genes_to_keep = genes_to_keep

    n_sibling_pairs = len(idx_values)

    chunk_size = (max(1, min(1000000//n_genes_to_keep, n_sibling_pairs)),
                  n_genes_to_keep)

    this_bounds = (idx_values[0], idx_values[-1]+1)

    with h5py.File(tmp_path, 'w') as out_file:

        out_file.create_dataset(
            'ranked_list',
            shape=(n_sibling_pairs, n_genes_to_keep),
            dtype=int,
            chunks=chunk_size,
            compression='gzip')

        if keep_all_stats:
            out_file.create_dataset(
                'validity',
                shape=(n_sibling_pairs, n_genes),
                dtype=bool,
                chunks=chunk_size,
                compression='gzip')

            out_file.create_dataset(
                'scores',
                shape=(n_sibling_pairs, n_genes),
                dtype=float,
                chunks=chunk_size)

    t0 = time.time()
    if output_lock is None:
        output_lock = DummyLock()

    ranked_list_buffer = np.zeros((flush_every, n_genes_to_keep), dtype=int)

    if keep_all_stats:
        validity_buffer = np.zeros((flush_every, n_genes), dtype=bool)
        score_buffer = np.zeros((flush_every, n_genes), dtype=float)

    buffer_idx = 0
    ct = 0
    for idx in idx_values:
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        node1 = sibling_pair[1]
        node2 = sibling_pair[2]

        pop1 = tree_as_leaves[level][node1]
        pop2 = tree_as_leaves[level][node2]
        (scores,
         validity) = score_differential_genes(
                         leaf_population_1=pop1,
                         leaf_population_2=pop2,
                         precomputed_stats=cluster_stats,
                         gt1_threshold=gt1_threshold,
                         gt0_threshold=gt0_threshold)

        ranked_list = rank_genes(
                         scores=scores,
                         validity=validity)

        if genes_to_keep is not None:
            ranked_list = ranked_list[:n_genes_to_keep]

        ranked_list_buffer[buffer_idx, :] = ranked_list

        if keep_all_stats:
            score_buffer[buffer_idx, :] = scores
            validity_buffer[buffer_idx, :] = validity

        buffer_idx += 1
        ct += 1
        if buffer_idx == flush_every or idx==idx_values[-1]:
            with h5py.File(tmp_path, 'a') as out_file:
                out_idx1 = idx+1-min(idx_values)
                out_idx0 = out_idx1-buffer_idx
                out_file['ranked_list'][out_idx0:out_idx1, :] = ranked_list_buffer[:buffer_idx, :]
                if keep_all_stats:
                    out_file['scores'][out_idx0:out_idx1, :] = score_buffer[:buffer_idx, :]
                    out_file['validity'][out_idx0:out_idx1, :] = validity_buffer[:buffer_idx, :]
                buffer_idx = 0

            n_bytes = file_size_in_bytes(tmp_path)
            msg = f"file size {n_bytes/(1024**3)} GB"
            print_timing(
                t0=t0,
                i_chunk=ct+1,
                tot_chunks=len(idx_values),
                unit='hr',
                msg=msg)

    # write this output from the temporary file to
    # the final output file
    if keep_all_stats:
        key_list = ('ranked_list', 'scores', 'validity')
    else:
        key_list = ('ranked_list', )

    with output_lock:
        d_write = max(1, 10000000//n_genes_to_keep)
        with h5py.File(tmp_path, 'r') as in_file:
            with h5py.File(output_path, 'a') as out_file:
                for i0 in range(this_bounds[0], this_bounds[1], d_write):
                    i1 = min(this_bounds[1], i0+d_write)
                    for k in key_list:
                        out_file[k][i0:i1] = in_file[k][i0-this_bounds[0]:
                                                        i1-this_bounds[0]]
    tmp_path.unlink()


def read_precomputed_stats(
        precomputed_stats_path):
    """
    Read in the precomputed stats file at
    precomputed_stats path and return a dict

    precomputed_stats:
        'gene_names': list of gene names
        'cluster_stats': Dict mapping leaf node name to
            'n_cells'
            'sum'
            'sumsq'
            'gt0'
            'gt1'
    """

    precomputed_stats = dict()

    with h5py.File(precomputed_stats_path, 'r') as in_file:

        precomputed_stats['gene_names'] = json.loads(
            in_file['col_names'][()].decode('utf-8'))

        row_lookup = json.loads(
            in_file['cluster_to_row'][()].decode('utf-8'))

        n_cells = in_file['n_cells'][()]
        sum_arr = in_file['sum'][()]
        sumsq_arr = in_file['sumsq'][()]
        gt0_arr = in_file['gt0'][()]
        gt1_arr = in_file['gt1'][()]

    cluster_stats = dict()
    for leaf_name in row_lookup:
        idx = row_lookup[leaf_name]
        this = dict()
        this['n_cells'] = n_cells[idx]
        this['sum'] = sum_arr[idx, :]
        this['sumsq'] = sumsq_arr[idx, :]
        this['gt0'] = gt0_arr[idx, :]
        this['gt1'] = gt1_arr[idx, :]
        cluster_stats[leaf_name] = this

    precomputed_stats['cluster_stats'] = cluster_stats
    return precomputed_stats


def score_differential_genes(
        leaf_population_1,
        leaf_population_2,
        precomputed_stats,
        gt1_threshold=0,
        gt0_threshold=1):
    """
    Rank genes according to their ability to differentiate between
    two populations fo cells.

    Parameters
    ----------
    leaf_population_1/2:
        Lists of names of the leaf nodes (e.g. clusters) of the cell
        taxonomy making up the two populations to compare.

    precomputed_stats:
        Dict mapping leaf node name to
            'n_cells'
            'sum'
            'sumsq'
            'gt0'
            'gt1'

    gt1_thresdhold/gt0_threshold:
        Number of cells that must express above 0/1 in order to be
        considered a valid differential gene.

    Returns
    -------
    score:
        np.ndarray of numerical scores indicating how good a gene
        is a s differentiator; larger values mean it is a better
        differentiator

    validity_mask:
        np.ndarray of booleans that is a mask for whether or not
        the gene passed the gt1, gt0 thresholds
    """

    stats_1 = aggregate_stats(
                leaf_population=leaf_population_1,
                precomputed_stats=precomputed_stats,
                gt0_threshold=gt0_threshold,
                gt1_threshold=gt1_threshold)

    stats_2 = aggregate_stats(
                leaf_population=leaf_population_2,
                precomputed_stats=precomputed_stats,
                gt0_threshold=gt0_threshold,
                gt1_threshold=gt1_threshold)

    score = diffexp_score(
                mean1=stats_1['mean'],
                var1=stats_1['var'],
                n1=stats_1['n_cells'],
                mean2=stats_2['mean'],
                var2=stats_2['var'],
                n2=stats_2['n_cells'])

    validity_mask = np.logical_or(
                        stats_1['mask'],
                        stats_2['mask'])

    return score, validity_mask


def diffexp_score(
        mean1,
        var1,
        n1,
        mean2,
        var2,
        n2):
    """
    Parameters (np.ndarrays of shape (n_genes, ))
    ---------------------------------------------
    mean1 -- mean gene expression values in pop1
    var1 -- variance of gene expression values in pop1
    n1 -- number of cells in pop1
    mean2 -- mean gene expression values in pop2
    var2 -- variance of gene expression values in pop2
    n2 -- number of cells in pop2

    Returns
    -------
    A np.ndarray of shape (n_genes,) representing
    the differential score of each gene at distinguishing
    between these two populations
    """

    (tt_stat,
     tt_nunu,
     pvalues) = welch_t_test(
                    mean1=mean1,
                    var1=var1,
                    n1=n1,
                    mean2=mean2,
                    var2=var2,
                    n2=n2)

    pvalues = correct_ttest(pvalues)
    with np.errstate(divide='ignore'):
        score = -1.0*np.log(pvalues)
    return score


def aggregate_stats(
       leaf_population,
       precomputed_stats,
       gt0_threshold=1,
       gt1_threshold=0):
    """
    Parameters
    ----------
    leaf_population:
        List of names of the leaf nodes (e.g. clusters) of the cell
        taxonomy making up the two populations to compare.

    precomputed_stats:
        Dict mapping leaf node name to
            'n_cells'
            'sum'
            'sumsq'
            'gt0'
            'gt1'

    gt1_thresdhold/gt0_threshold:
        Number of cells that must express above 0/1 in order to be
        considered a valid differential gene.

    Returns
    -------
    Dict with
        'mean' -- mean value of all gene expression
        'var' -- variance of all gene expression
        'mask' -- boolean mask of genes that pass thresholds
        'n_cells' -- number of cells in the population
    """
    n_genes = len(precomputed_stats[leaf_population[0]]['sum'])

    sum_arr = np.zeros(n_genes, dtype=float)
    sumsq_arr = np.zeros(n_genes, dtype=float)
    gt0 = np.zeros(n_genes, dtype=int)
    gt1 = np.zeros(n_genes, dtype=int)
    n_cells = 0

    for leaf_node in leaf_population:
        these_stats = precomputed_stats[leaf_node]

        n_cells += these_stats['n_cells']
        sum_arr += these_stats['sum']
        sumsq_arr += these_stats['sumsq']
        gt0 += these_stats['gt0']
        gt1 += these_stats['gt1']

    mu = sum_arr/n_cells
    var = (sumsq_arr-sum_arr**2/n_cells)/max(1, n_cells-1)

    mask = np.logical_and(
                gt0 >= gt0_threshold,
                gt1 >= gt1_threshold)

    return {'mean': mu,
            'var': var,
            'mask': mask,
            'n_cells': n_cells}


def rank_genes(
        scores,
        validity):
    """
    Parameters
    ----------
    scores:
        An np.ndarray of floats; the diffexp scores
        of each gene
    validity:
        An np.ndarray of booleans; a flag indicating
        if the gene passed all of the validity tests

    Returns
    -------
    ranked_list:
        An np.ndarray of ints. ranked_list[0] is the index
        of the best discriminator. ranked_list[-1] is the
        index of the worst discriminator.

        Valid genes are all ranked before invalid genes.
    """
    max_score = scores.max()
    joint_stats = np.copy(scores)
    joint_stats[validity] += max_score+1.0
    sorted_dex = np.argsort(-1.0*joint_stats)
    return sorted_dex
