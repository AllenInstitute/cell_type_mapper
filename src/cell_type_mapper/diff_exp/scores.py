import numpy as np

from cell_type_mapper.utils.stats_utils import (
    welch_t_test,
    correct_ttest,
    approx_correct_ttest)

from cell_type_mapper.diff_exp.score_utils import (
    q_score_from_pij,
    pij_from_stats)


def score_differential_genes(
        node_1,
        node_2,
        precomputed_stats,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        log2_fold_th=1.0,
        q1_min_th=0.1,
        qdiff_min_th=0.1,
        log2_fold_min_th=0.8,
        n_cells_min=2,
        boring_t=None,
        big_nu=None,
        exact_penetrance=False,
        n_valid=30,
        n_valid_min=10,
        valid_gene_idx=None):
    """
    Rank genes according to their ability to differentiate between
    two populations of cells.

    Parameters
    ----------
    node_1/2:
        Names of the leaf nodes (e.g. clusters) of the cell
        taxonomy making up the two populations to compare.

    precomputed_stats:
        Dict mapping leaf node name to
            'n_cells'
            'mean'
            'var'
            'n_cells'
            'ge1'

    p_th/q1_th/qdiff_th:
        Thresholds for determining if the gene is a differentially
        expressed gene (see Notes below)

    log2_fold_th:
        Genes must have a log2(fold changes) > log2_fold_th
        between the two populations to be considered a
        marker gene.

    q1_min_th/qdiff_min_th/log2_fold_min_th:
        Minimum thresholds on q1, qdiff, and log2_fold
        below which genes will not be considered markers,
        even when using the approximate penetrance tests.

    n_cells_min:
        If either node has fewer cells than this, return
        placeholder answer in which no genes are markers.

    boring_t:
       If not None, values of the t-test statistic must be
       outisde the range (-boring_t, boring_t) to be considered
       "interesting." "Uninteresting" values will be given a CDF
       value of 0.5

    big_nu:
        If not None, Student t-test distributions with more degrees
        of freedom than big_nu will be approximated with the
        normal distribution.

    exact_penetrance:
        If True, hold marker genes to the exact penetrance criteria;
        if not, use an approximation to assure there are a minimum
        number of valid marker genes for the cluster pair.

    n_valid:
        Number of markers to try for when using approximate
        penetrance test

    n_valid_min:
        If the number of markers is less than this and exact_penetrance
        is False, re-rerun marker finding with a slightly more relaxed
        criterion (namely, take passage of p-value test into consideration
        when applying penetrance test).

    valid_gene_idx:
        Optional numpy array of indexes of genes that can be considered
        markers. If not None, genes outside of this set will have their
        p_ij values set arbitrarily to zero so that they will fail the
        penetrance test.

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

    stats_1 = precomputed_stats[node_1]
    stats_2 = precomputed_stats[node_2]

    n_genes = len(stats_1['mean'])
    if stats_1['n_cells'] < n_cells_min or stats_2['n_cells'] < n_cells_min:
        return (np.zeros(n_genes, dtype=float),
                np.zeros(n_genes, dtype=bool),
                np.zeros(n_genes, dtype=bool))

    pvalues = diffexp_p_values_from_stats(
            node_1=node_1,
            node_2=node_2,
            precomputed_stats=precomputed_stats,
            p_th=p_th,
            big_nu=big_nu,
            boring_t=boring_t)

    pvalue_valid = (pvalues < p_th)

    keep_going = True
    n_iteration = 0

    while keep_going:
        n_iteration += 1
        penetrance_mask = penetrance_from_stats(
            node_1=node_1,
            node_2=node_2,
            precomputed_stats=precomputed_stats,
            q1_th=q1_th,
            q1_min_th=q1_min_th,
            qdiff_th=qdiff_th,
            qdiff_min_th=qdiff_min_th,
            log2_fold_th=log2_fold_th,
            log2_fold_min_th=log2_fold_min_th,
            valid_gene_idx=valid_gene_idx,
            exact_penetrance=exact_penetrance,
            n_valid=n_valid)

        validity_mask = np.logical_and(
            pvalue_valid,
            penetrance_mask)

        # If not enough markers were found, try setting
        # valid_gene_idx so that it takes account of genes
        # that we know will pass the p-value test.
        # However, we will only make one more pass through.

        if validity_mask.sum() >= n_valid_min:
            keep_going = False
        if exact_penetrance:
            keep_going = False
        if n_iteration > 1:
            keep_going = False

        if keep_going:
            if valid_gene_idx is None:
                gene_mask = np.ones(
                        stats_1['mean'].shape,
                        dtype=bool)
            else:
                gene_mask = np.zeros(
                        stats_1['mean'].shape,
                        dtype=bool)
                gene_mask[valid_gene_idx] = True
            valid_gene_idx = np.where(
                np.logical_and(
                    gene_mask,
                    pvalue_valid))[0]

    up_mask = np.zeros(stats_1["mean"].shape, dtype=np.uint8)
    up_mask[stats_2["mean"] > stats_1["mean"]] = 1

    return -1.0*np.log(pvalues), validity_mask, up_mask


def diffexp_p_values_from_stats(
        node_1,
        node_2,
        precomputed_stats,
        p_th,
        big_nu=None,
        boring_t=None):
    """
    Parameters
    ----------
    node_1/2:
        Names of the leaf nodes (e.g. clusters) of the cell
        taxonomy making up the two populations to compare.

    precomputed_stats:
        Dict mapping leaf node name to
            'n_cells'
            'mean'
            'var'
            'n_cells'
            'ge1'

    p_th:
        Threshold on acceptable p-values

    big_nu:
        If not None, Student t-test distributions with more degrees
        of freedom than big_nu will be approximated with the
        normal distribution.

    boring_t:
       If not None, values of the t-test statistic must be
       outisde the range (-boring_t, boring_t) to be considered
       "interesting." "Uninteresting" values will be given a CDF
       value of 0.5

    Returns
    -------
    A np.ndarray of shape (n_genes,) representing
    the corrected p_values of the genes as markers
    """
    stats_1 = precomputed_stats[node_1]
    stats_2 = precomputed_stats[node_2]

    pvalues = diffexp_p_values(
                mean1=stats_1['mean'],
                var1=stats_1['var'],
                n1=stats_1['n_cells'],
                mean2=stats_2['mean'],
                var2=stats_2['var'],
                n2=stats_2['n_cells'],
                boring_t=boring_t,
                big_nu=big_nu,
                p_th=p_th)

    return pvalues


def diffexp_p_values(
        mean1,
        var1,
        n1,
        mean2,
        var2,
        n2,
        boring_t=None,
        big_nu=None,
        p_th=None):
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

    p_th:
       If not None, p-values above this threshold will not be
       passed to the correct_ttest function (since they
       are already going to fail a threshold cut)

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

    if p_th is None:
        pvalues = correct_ttest(pvalues)
    else:
        pvalues = approx_correct_ttest(pvalues, p_th=p_th)

    return pvalues


def penetrance_from_stats(
        node_1,
        node_2,
        precomputed_stats,
        q1_th,
        q1_min_th,
        qdiff_th,
        qdiff_min_th,
        log2_fold_th,
        log2_fold_min_th,
        valid_gene_idx,
        exact_penetrance,
        n_valid):
    """
    Return an (n_genes,) mask of genes that pass the penetrance
    tests.

    Parameters
    ----------
    node_1/2:
        Names of the leaf nodes (e.g. clusters) of the cell
        taxonomy making up the two populations to compare.

    precomputed_stats:
        Dict mapping leaf node name to
            'n_cells'
            'mean'
            'var'
            'n_cells'
            'ge1'

    p_th/q1_th/qdiff_th:
        Thresholds for determining if the gene is a differentially
        expressed gene (see Notes below)

    log2_fold_th:
        Genes must have a log2(fold changes) > log2_fold_th
        between the two populations to be considered a
        marker gene.

    q1_min_th/qdiff_min_th/log2_fold_min_th:
        Minimum thresholds on q1, qdiff, and log2_fold
        below which genes will not be considered markers,
        even when using the approximate penetrance tests.

    exact_penetrance:
        If True, hold marker genes to the exact penetrance criteria;
        if not, use an approximation to assure there are a minimum
        number of valid marker genes for the cluster pair.

    n_valid:
        Number of markers to try for when using approximate
        penetrance test

    valid_gene_idx:
        Optional numpy array of indexes of genes that can be considered
        markers. If not None, genes outside of this set will have their
        p_ij values set arbitrarily to zero so that they will fail the
        penetrance test.

    Returns
    -------
    An (n_genes,) array of booleans indicating which genes
    passed the penetrance tests.
    """

    (pij_1,
     pij_2,
     log2_fold) = pij_from_stats(
             cluster_stats=precomputed_stats,
             node_1=node_1,
             node_2=node_2)

    if valid_gene_idx is not None:
        invalid_mask = np.zeros(pij_1.shape, dtype=bool)
        invalid_mask[valid_gene_idx] = True
        invalid_mask = np.logical_not(invalid_mask)
        pij_1[invalid_mask] = -1.0
        pij_2[invalid_mask] = -1.0
        log2_fold[invalid_mask] = -1.0

    penetrance_mask = penetrance_tests(
        pij_1=pij_1,
        pij_2=pij_2,
        log2_fold=log2_fold,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        log2_fold_th=log2_fold_th,
        exact=exact_penetrance,
        q1_min_th=q1_min_th,
        qdiff_min_th=qdiff_min_th,
        log2_fold_min_th=log2_fold_min_th,
        n_valid=n_valid)
    return penetrance_mask


def penetrance_tests(
        pij_1,
        pij_2,
        log2_fold,
        q1_th,
        qdiff_th,
        log2_fold_th,
        exact=False,
        q1_min_th=0.1,
        qdiff_min_th=0.1,
        log2_fold_min_th=0.8,
        n_valid=30):
    """
    Perform penetrance test on marker genes

    Parameters
    ----------
    pij_1:
        (n_genes,) array representing what fraction of
        cells in cluster one are expressed > 1 for the gene
    pij_2:
        ditto for cluster 2
    log2_fold:
        (n_genes,) array of the log2(fold_change) (absolute
        magnitude) of each gene between the two clusters
    q1_th:
        At least one cluster must have a penetrance
        greater than this to pass
    qdiff_th:
        differential penetrance must be greater than
        this to pass
    log2_fold_th:
        Minimum acceptable threshold of log2 fold change
    exact:
        If True, only pass genes that exactly meet the
        criteria defined by q1_th and qdiff_th. Otherwise,
        use an approximation to make sure there are at least
        30 valid marker genes.
    q1_min_th, qdiff_min_th, log2_fold_min_th:
        The minimum thresholds for q1, qdiff, log2 fold change
        for which a gene will automatically be considered an
        invalid marker in the approximate penetrance test. This
        is not used if exact=True.
    n_valid:
        number of markers to try for when using approximate test

    Returns
    -------
    penentrance_mask:
        (n_genes,) array of booleans that pass both tests
    """

    (q1_score,
     qdiff_score) = q_score_from_pij(
                         pij_1=pij_1,
                         pij_2=pij_2)

    if exact:
        raw_penetrance = exact_penetrance_test(
            q1_score=q1_score,
            qdiff_score=qdiff_score,
            q1_th=q1_th,
            qdiff_th=qdiff_th)

        fold_valid = (log2_fold > log2_fold_th)
        return np.logical_and(fold_valid, raw_penetrance)

    return approx_penetrance_test(
        q1_score=q1_score,
        qdiff_score=qdiff_score,
        log2_fold=log2_fold,
        q1_th=q1_th,
        qdiff_th=qdiff_th,
        log2_fold_th=log2_fold_th,
        q1_min_th=q1_min_th,
        qdiff_min_th=qdiff_min_th,
        log2_fold_min_th=log2_fold_min_th,
        n_valid=n_valid)


def exact_penetrance_test(
        q1_score,
        qdiff_score,
        q1_th,
        qdiff_th):

    q1_valid = (q1_score > q1_th)
    qdiff_valid = (qdiff_score > qdiff_th)
    return np.logical_and(q1_valid, qdiff_valid)


def approx_penetrance_test(
        q1_score,
        qdiff_score,
        log2_fold,
        q1_th,
        q1_min_th,
        qdiff_th,
        qdiff_min_th,
        log2_fold_th,
        log2_fold_min_th,
        n_valid=30):
    """
    Use an approximate cut on q1, qdiff to set genes as valid
    markers if they come close to meeting the penetrance criteria.
    """

    n_valid = min(n_valid, len(q1_score))

    distances = penetrance_parameter_distance(
        q1_score=q1_score,
        qdiff_score=qdiff_score,
        log2_fold=log2_fold,
        q1_th=q1_th,
        q1_min_th=q1_min_th,
        qdiff_th=qdiff_th,
        qdiff_min_th=qdiff_min_th,
        log2_fold_th=log2_fold_th,
        log2_fold_min_th=log2_fold_min_th)

    distance_sq = distances['true']
    qdiff_dist = distances['qdiff']
    q1_dist = distances['q1']
    fold_dist = distances['fold']

    # find the genes that really do meet the criteria
    eps = 1.0e-10
    absolutely_valid = (distance_sq < eps)

    # if not enough genes really meet the criteria, add
    # the next best approximations (failing out any genes
    # that are in violation of q1_min_th and qdiff_min_th
    if absolutely_valid.sum() >= n_valid:
        valid = absolutely_valid
    else:

        qdiff_dex = np.argsort(qdiff_dist)
        q1_dex = np.argsort(q1_dist)
        fold_dex = np.argsort(fold_dist)

        cutoff = min(q1_dist[q1_dex[n_valid-1]],
                     qdiff_dist[qdiff_dex[n_valid-1]],
                     fold_dist[fold_dex[n_valid-1]])

        to_use = set(np.where(qdiff_dist <= cutoff)[0])
        to_use = to_use.union(
            set(np.where(q1_dist <= cutoff)[0]))
        to_use = to_use.union(
            set(np.where(fold_dist <= cutoff)[0]))

        to_use = np.array(list(to_use))

        valid = np.zeros(len(absolutely_valid), dtype=bool)
        valid[to_use] = True
        valid[distances['invalid']] = False

    return valid


def penetrance_parameter_distance(
        q1_score,
        qdiff_score,
        log2_fold,
        q1_th,
        q1_min_th,
        qdiff_th,
        qdiff_min_th,
        log2_fold_th,
        log2_fold_min_th):
    """
    Return a dict with
        {'true': true distance_sq
         'wgt': weighted distance_sq
         'q1': q1_distance_sq
         'qdiff': qdiff_distance_sq
         'fold': fold_distance_sq,
         'invalid': mask marking genes that are invalid by min_th}
    """

    if len(q1_score) != len(qdiff_score) or len(q1_score) != len(log2_fold):
        raise RuntimeError(
            "q1_score, qdiff_score, and log2_fold must all have same shape; "
            f"you have {len(q1_score)}, {len(qdiff_score)}, {len(log2_fold)}")

    if q1_th <= q1_min_th:
        raise RuntimeError(
            "q1_th must be > q1_min_th; you have "
            f"q1_th={q1_th:.2e}, q1_min_th={q1_min_th}")

    if qdiff_th <= qdiff_min_th:
        raise RuntimeError(
            "qdiff_th must be > qdiff_min_th; you have "
            f"qdiff_th={qdiff_th:.2e}, qdiff_min_th={qdiff_min_th}")

    if log2_fold_th <= log2_fold_min_th:
        raise RuntimeError(
            "log2_fold_th must be > log2_fold_min_th; you have "
            f"log2_fold_th={log2_fold_th:.2e}, "
            f"log2_fold_min_th={log2_fold_min_th}")

    q1_term = (q1_score-q1_th)**2
    q1_term[q1_score > q1_th] = 0.0

    qdiff_term = (qdiff_score-qdiff_th)**2
    qdiff_term[qdiff_score > qdiff_th] = 0.0

    fold_term = (log2_fold-log2_fold_th)**2
    fold_term[log2_fold > log2_fold_th] = 0.0

    distance_sq = qdiff_term+q1_term+fold_term

    # alternatively upweight the metrics so that one
    # does not predominate

    qdiff_dist = 1.5*qdiff_term+q1_term+fold_term
    q1_dist = qdiff_term+1.5*q1_term+fold_term
    fold_dist = qdiff_term+q1_term+1.5*fold_term

    invalid = np.logical_or(
            q1_score < q1_min_th,
            np.logical_or(
                qdiff_score < qdiff_min_th,
                log2_fold < log2_fold_min_th))

    # alter distances so that we do not choose
    # genes that are going to be labeled automatically
    # invalid anyway
    bad_dist = max(
        qdiff_dist.max(),
        q1_dist.max(),
        fold_dist.max()) + 100.0

    qdiff_dist[invalid] = bad_dist
    q1_dist[invalid] = bad_dist
    fold_dist[invalid] = bad_dist

    wgt = qdiff_dist
    wgt = np.where(
        wgt < q1_dist,
        wgt,
        q1_dist)
    wgt = np.where(
        wgt < fold_dist,
        wgt,
        fold_dist)

    return {
        'true': distance_sq,
        'q1': q1_dist,
        'qdiff': qdiff_dist,
        'fold': fold_dist,
        'wgt': wgt,
        'invalid': invalid
    }


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
