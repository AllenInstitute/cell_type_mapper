import json
import h5py
import numpy as np

from hierarchical_mapping.utils.taxonomy_utils import (
    convert_tree_to_leaves)

from hierarchical_mapping.utils.stats_utils import (
    welch_t_test,
    correct_ttest)


def score_all_taxonomy_pairs(
        precomputed_stats_path,
        taxonomy_tree,
        gt1_threshold=0,
        gt0_threshold=1):
    """
    Create differential expression scores and validity masks
    for differential genes between all relevant pairs in a
    taxonomy.

    Parameters
    ----------
    precomputed_stats_path:
        Path to HDF5 file containing precomputed stats for leaf nodes

    taxonomy_tree:
        Dict encoding the taxonomy tree (created when we create the
        contiguous zarr file and stored in that file's metadata.json)

    gt1_thresdhold/gt0_threshold:
        Number of cells that must express above 0/1 in order to be
        considered a valid differential gene.

    Returns
    --------
    A dict structured like
        level ->
            node1 ->
                node2 ->
                    score: score of genes comparing node1 to node2
                    validity: mask of genes that pass the gt thresholds

    * keys are sorted so node1 always precedes node2 alphabetically.
    """
    hierarchy = taxonomy_tree['hierarchy']
    tree_as_leaves = convert_tree_to_leaves(taxonomy_tree)

    precomputed_stats = read_precomputed_stats(
           precomputed_stats_path)

    results = dict()
    for level in hierarchy:
        this_level = dict()
        node_list = list(taxonomy_tree[level].keys())
        node_list.sort()
        for i1 in range(len(node_list)):
            node1 = node_list[i1]
            pop1 = tree_as_leaves[level][node1]
            this_level[node1] = dict()
            for i2 in range(i1+1, len(node_list), 1):
                node2 = node_list[i2]
                pop2 = tree_as_leaves[level][node2]
                (score,
                 validity) = score_differential_genes(
                                 leaf_population_1=pop1,
                                 leaf_population_2=pop2,
                                 precomputed_stats=precomputed_stats,
                                 gt1_threshold=gt1_threshold,
                                 gt0_threshold=gt0_threshold)
                this_level[node1][node2] = {'score': score,
                                            'validity': validity}
        results[level] = this_level
    return results


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
