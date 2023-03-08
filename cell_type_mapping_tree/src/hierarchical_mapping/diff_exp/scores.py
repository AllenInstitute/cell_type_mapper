import json
import h5py
import numpy as np
import time

from hierarchical_mapping.utils.utils import (
    print_timing)

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves,
    get_siblings)

from hierarchical_mapping.utils.stats_utils import (
    welch_t_test,
    correct_ttest)


def score_all_taxonomy_pairs(
        precomputed_stats_path,
        taxonomy_tree,
        output_path,
        gt1_threshold=0,
        gt0_threshold=1,
        flush_every=1000):
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

    hierarchy = taxonomy_tree['hierarchy']

    siblings = get_siblings(taxonomy_tree)

    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)

    precomputed_stats = read_precomputed_stats(
           precomputed_stats_path)

    n_sibling_pairs = len(siblings)
    n_genes = len(precomputed_stats[list(precomputed_stats.keys())[0]]['sum'])

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


    chunk_size = (max(1, min(1000000//n_genes, n_sibling_pairs)),
                  n_genes)

    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset(
            'pair_to_idx',
            data=json.dumps(pair_to_idx_out).encode('utf-8'))

        out_file.create_dataset(
            'validity',
            shape=(n_sibling_pairs, n_genes),
            dtype=bool,
            chunks=chunk_size,
            compression='gzip')

        out_file.create_dataset(
            'ranked_list',
            shape=(n_sibling_pairs, n_genes),
            dtype=int,
            chunks=chunk_size,
            compression='gzip')

        out_file.create_dataset(
            'scores',
            shape=(n_sibling_pairs, n_genes),
            dtype=float,
            chunks=chunk_size)

    print("starting to score")
    t0 = time.time()

    idx_values = list(idx_to_pair.keys())
    idx_values.sort()

    ranked_list_buffer = np.zeros((flush_every, n_genes), dtype=int)
    validity_buffer = np.zeros((flush_every, n_genes), dtype=bool)
    score_buffer = np.zeros((flush_every, n_genes), dtype=float)

    with h5py.File(output_path, 'a') as out_file:
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
                             precomputed_stats=precomputed_stats,
                             gt1_threshold=gt1_threshold,
                             gt0_threshold=gt0_threshold)

            ranked_list = rank_genes(
                             scores=scores,
                             validity=validity)

            ranked_list_buffer[buffer_idx, :] = ranked_list
            score_buffer[buffer_idx, :] = scores
            validity_buffer[buffer_idx, :] = validity
            buffer_idx += 1
            ct += 1
            if buffer_idx == flush_every or idx==idx_values[-1]:
                out_idx1 = idx+1
                out_idx0 = out_idx1-buffer_idx
                out_file['ranked_list'][out_idx0:out_idx1, :] = ranked_list_buffer[:buffer_idx, :]
                out_file['scores'][out_idx0:out_idx1, :] = score_buffer[:buffer_idx, :]
                out_file['validity'][out_idx0:out_idx1, :] = validity_buffer[:buffer_idx, :]
                buffer_idx = 0

                print_timing(
                    t0=t0,
                    i_chunk=ct+1,
                    tot_chunks=n_sibling_pairs,
                    unit='hr')


def read_precomputed_stats(
        precomputed_stats_path):
    """
    Read in the precomputed stats file at
    precomputed_stats path and return a dict

    precomputed_stats:
        Dict mapping leaf node name to
            'n_cells'
            'sum'
            'sumsq'
            'gt0'
            'gt1'
    """
    with h5py.File(precomputed_stats_path, 'r') as in_file:
        row_lookup = json.loads(
            in_file['cluster_to_row'][()].decode('utf-8'))

        n_cells = in_file['n_cells'][()]
        sum_arr = in_file['sum'][()]
        sumsq_arr = in_file['sumsq'][()]
        gt0_arr = in_file['gt0'][()]
        gt1_arr = in_file['gt1'][()]

    precomputed_stats = dict()
    for leaf_name in row_lookup:
        idx = row_lookup[leaf_name]
        this = dict()
        this['n_cells'] = n_cells[idx]
        this['sum'] = sum_arr[idx, :]
        this['sumsq'] = sumsq_arr[idx, :]
        this['gt0'] = gt0_arr[idx, :]
        this['gt1'] = gt1_arr[idx, :]
        precomputed_stats[leaf_name] = this

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
