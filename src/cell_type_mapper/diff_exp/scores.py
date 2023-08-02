import json
import h5py
import numpy as np
import warnings

from cell_type_mapper.utils.stats_utils import (
    welch_t_test,
    correct_ttest)


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

    return this_cluster_stats, this_tree_as_leaves


def read_precomputed_stats(
        precomputed_stats_path):
    """
    Read in the precomputed stats file at
    precomputed_stats path and return a dict

    precomputed_stats:
        'gene_names': list of gene names
        'cluster_stats': Dict mapping leaf node name to
            'n_cells'
            'sum'  -- units of log2(CPM+1)
            'sumsq' -- units of log2(CPM+1)
            'gt0'
            'gt1'
            'ge1'
    """

    precomputed_stats = dict()
    raw_data = dict()
    with h5py.File(precomputed_stats_path, 'r') as in_file:

        precomputed_stats['gene_names'] = json.loads(
            in_file['col_names'][()].decode('utf-8'))

        row_lookup = json.loads(
            in_file['cluster_to_row'][()].decode('utf-8'))

        for k in ('n_cells', 'sum', 'sumsq', 'gt0', 'gt1', 'ge1'):
            if k in in_file:
                raw_data[k] = in_file[k][()]

    cluster_stats = dict()
    for leaf_name in row_lookup:
        idx = row_lookup[leaf_name]
        this = dict()
        if 'n_cells' in raw_data:
            this['n_cells'] = raw_data['n_cells'][idx]
        for k in ('sum', 'sumsq', 'gt0', 'gt1', 'ge1'):
            if k in raw_data:
                this[k] = raw_data[k][idx, :]
        cluster_stats[leaf_name] = this

    precomputed_stats['cluster_stats'] = cluster_stats
    return precomputed_stats


def score_differential_genes(
        leaf_population_1,
        leaf_population_2,
        precomputed_stats,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        fold_change=2.0,
        boring_t=None,
        big_nu=None):
    """
    Rank genes according to their ability to differentiate between
    two populations of cells.

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

    p_th/q1_th/qdiff_th:
        Thresholds for determining if the gene is a differentially
        expressed gene (see Notes below)

    fold_change:
        Genes must have a fold changes > fold_change between the
        two populations to be considered a marker gene.

    boring_t:
       If not None, values of the t-test statistic must be
       outisde the range (-boring_t, boring_t) to be considered
       "interesting." "Uninteresting" values will be given a CDF
       value of 0.5

    big_nu:
        If not None, Student t-test distributions with more degrees
        of freedom than big_nu will be approximated with the
        normal distribution.

    Returns
    -------
    score:
        np.ndarray of numerical scores indicating how good a gene
        is a s differentiator; larger values mean it is a better
        differentiator

    validity_mask:
        np.ndarray of booleans that is a mask for whether or not
        the gene passes the criteria for being a marker gene

    up_mask:
        Array of unsigned integers that is (n_genes,) in size.
        Will be 0 for genes that are more prevalent in leaf_population_1
        and 1 for genes that are mre prevalent in leaf_population_2

    Notes
    -----
    'sum' and 'sumsq' are in units of log2(CPM+1)

    Marker gene criteria (from Tasic et al. 2018):

        adjusted p_value < p_th

        more than twofold expression change between clusters

        define P_ij as the fraction of cells in cluster j expressing gene
        i at greater than 1CPM
            P_ij > q1_th for at least one cluster (the up-regulated cluster)
            (P_i1j-Pi2j)/max(P_i1j, P_i2j) > qdiff_th
    """

    stats_1 = aggregate_stats(
                leaf_population=leaf_population_1,
                precomputed_stats=precomputed_stats)

    stats_2 = aggregate_stats(
                leaf_population=leaf_population_2,
                precomputed_stats=precomputed_stats)

    pvalues = diffexp_p_values(
                mean1=stats_1['mean'],
                var1=stats_1['var'],
                n1=stats_1['n_cells'],
                mean2=stats_2['mean'],
                var2=stats_2['var'],
                n2=stats_2['n_cells'],
                boring_t=boring_t,
                big_nu=big_nu)

    pvalue_valid = (pvalues < p_th)

    pij_1 = stats_1['ge1']/max(1, stats_1['n_cells'])
    pij_2 = stats_2['ge1']/max(1, stats_2['n_cells'])

    penetrance_mask = penetrance_tests(
        pij_1=pij_1,
        pij_2=pij_2,
        q1_th=q1_th,
        qdiff_th=qdiff_th)

    log2_fold = np.log2(fold_change)
    fold_valid = (np.abs(stats_1['mean']-stats_2['mean']) > log2_fold)

    validity_mask = np.logical_and(
        pvalue_valid,
        np.logical_and(
            penetrance_mask,
            fold_valid))

    up_mask = np.zeros(pij_1.shape, dtype=np.uint8)
    up_mask[stats_2["mean"] > stats_1["mean"]] = 1

    return -1.0*np.log(pvalues), validity_mask, up_mask


def diffexp_p_values(
        mean1,
        var1,
        n1,
        mean2,
        var2,
        n2,
        boring_t=None,
        big_nu=None):
    """
    Parameters (np.ndarrays of shape (n_genes, ))
    ---------------------------------------------
    mean1 -- mean gene expression values in pop1
    var1 -- variance of gene expression values in pop1
    n1 -- number of cells in pop1
    mean2 -- mean gene expression values in pop2
    var2 -- variance of gene expression values in pop2
    n2 -- number of cells in pop2

    boring_t:
       If not None, values of the t-test statistic must be
       outisde the range (-boring_t, boring_t) to be considered
       "interesting." "Uninteresting" values will be given a CDF
       value of 0.5

    big_nu:
        If not None, Student t-test distributions with more degrees
        of freedom than big_nu will be approximated with the
        normal distribution.

    Returns
    -------
    A np.ndarray of shape (n_genes,) representing
    the corrected p_values of the genes as markers

    Notes
    -----
    means and variances in input are in units of log2(CPM+1)
    """

    (_,
     _,
     pvalues) = welch_t_test(
                    mean1=mean1,
                    var1=var1,
                    n1=n1,
                    mean2=mean2,
                    var2=var2,
                    n2=n2,
                    boring_t=boring_t,
                    big_nu=big_nu)

    pvalues = correct_ttest(pvalues)
    return pvalues


def penetrance_tests(
        pij_1,
        pij_2,
        q1_th,
        qdiff_th):
    """
    Perform penetrance test on marker genes

    Parameters
    ----------
    pij_1:
        (n_genes,) array representing what fraction of
        cells in cluster one are expressed > 1 for the gene
    pij_2:
        ditto for cluster 2
    q1_th:
        At least one cluster must have a penetrance
        greater than this to pass
    qdiff_th:
        differential penetrance must be greater than
        this to pass

    Returns
    -------
    penentrance_mask:
        (n_genes,) array of booleans that pass both tests
    """
    q1_valid = np.logical_or(
        (pij_1 > q1_th),
        (pij_2 > q1_th))

    denom = np.where(pij_1 > pij_2, pij_1, pij_2)
    denom = np.where(denom > 0.0, denom, 1.0)
    qdiff_score = np.abs(pij_1-pij_2)/denom
    qdiff_valid = (qdiff_score > qdiff_th)
    return np.logical_and(qdiff_valid, q1_valid)


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

    Notes
    -----
    means and variances in input are in units of log2(CPM+1)
    """

    pvalues = diffexp_p_values(
        mean1=mean1,
        var1=var1,
        n1=n1,
        mean2=mean2,
        var2=var2,
        n2=n2)

    with np.errstate(divide='ignore'):
        score = -1.0*np.log(pvalues)
    return score


def aggregate_stats(
       leaf_population,
       precomputed_stats):
    """
    Parameters
    ----------
    leaf_population:
        List of names of the leaf nodes (e.g. clusters) of the cell
        taxonomy making up the two populations to compare.

    precomputed_stats:
        Dict mapping leaf node name to
            'n_cells'
            'sum' -- units of log2(CPM+1)
            'sumsq' -- units of log2(CPM+1)
            'gt0'
            'gt1'
            'ge1'

    Returns
    -------
    Dict with
        'mean' -- mean value of all gene expression
        'var' -- variance of all gene expression
        'n_cells' -- number of cells in the population

    Note
    -----
    output mean and var are in units of log2(CPM+1)

    Some historical versions of precomputed_stats files did
    not contain the 'ge1' column. If you are reading one of those,
    'ge1' will be returned as None.
    """
    n_genes = len(precomputed_stats[leaf_population[0]]['sum'])

    sum_arr = np.zeros(n_genes, dtype=float)
    sumsq_arr = np.zeros(n_genes, dtype=float)
    gt0 = np.zeros(n_genes, dtype=int)
    gt1 = np.zeros(n_genes, dtype=int)
    ge1 = np.zeros(n_genes, dtype=int)
    n_cells = 0
    has_ge1 = True

    for leaf_node in leaf_population:
        these_stats = precomputed_stats[leaf_node]

        n_cells += these_stats['n_cells']
        sum_arr += these_stats['sum']
        sumsq_arr += these_stats['sumsq']
        gt0 += these_stats['gt0']
        gt1 += these_stats['gt1']
        if 'ge1' in these_stats:
            ge1 += these_stats['ge1']
        else:
            has_ge1 = False

    mu = sum_arr/n_cells
    var = (sumsq_arr-sum_arr**2/n_cells)/max(1, n_cells-1)

    if not has_ge1:
        warnings.warn("precomputed stats file does not have 'ge1' data")
        ge1 = None

    return {'mean': mu,
            'var': var,
            'n_cells': n_cells,
            'gt0': gt0,
            'gt1': gt1,
            'ge1': ge1}


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
