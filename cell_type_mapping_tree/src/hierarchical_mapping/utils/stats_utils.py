import numpy as np
import scipy.stats as scipy_stats

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


def summary_stats_for_chunk(
        cell_x_gene: CellByGeneMatrix) -> dict:
    """
    Take a cell (rows) by gene (columns) expression
    matrix and return summary statistics needed to aggregate
    chunks for differential gene search.

    Parameters
    ----------
    cell_x_gene: CellByGeneMatrix
        normalization must be log2CPM

    Returns
    -------
    A dict of summary stats
        'n_cells' is the number of cells in this cluster
        'sum' is a (n_genes,) array summing the gene expression
        'sumsq' is a (n_genes,) array summy the square of gene expression
        'gt0' is a (n_genes,) array indicating
              how many cells have expression > 0 CPM
        'gt1' is a (n_genes,) array indicating
              how many cells have expression > 1 CPM
        'ge1' is a (n_genes,) array indicating
              how many cells have expression >= 1 CPM
    """
    if not cell_x_gene.normalization == 'log2CPM':
        raise RuntimeError(
            "cell_x_gene normalization is not log2CPM\n"
            f"is {cell_x_gene.normalization}")

    zero_cutoff = 0.0  # log2(CPM+1) with CPM=0
    one_cutoff = 1.0  # log2(CPM+1) with CPM=1
    eps = 1.0e-6  # for float comparisons

    result = dict()
    data = cell_x_gene.data
    result['n_cells'] = cell_x_gene.n_cells
    result['sum'] = data.sum(axis=0)
    result['sumsq'] = (data**2).sum(axis=0)
    result['gt0'] = (data > zero_cutoff).sum(axis=0)
    result['gt1'] = (data > one_cutoff).sum(axis=0)
    result['ge1'] = (data > one_cutoff-eps).sum(axis=0)
    return result


def welch_t_test(mean1, var1, n1, mean2, var2, n2):
    """
    Perform Welch's t-test on the gene expression values in
    two populations

    Parameters
    ----------
    mean1 -- mean gene expression values in pop1
    var1 -- variance of gene expression values in pop1
    n1 -- number of cells in pop1
    mean2 -- mean gene expression values in pop2
    var2 -- variance of gene expression values in pop2
    n2 -- number of cells in pop2

    Returns
    -------
    tt -- the t-test statistic value for each gene
    nu -- the number of degrees of freedom in the Student's t-distribution
          for each gene
    pval -- the p-value of the significance of the difference in gene
            expression distributions between pop1 and pop2
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        nu_num = var1/n1 + var2/n2
        denom = np.sqrt(nu_num)
        denom = np.where(denom > 0.0, denom, 1.0e-10)
        tt = (mean1-mean2)/denom
        nu_denom = ((var1**2)/(n1**3-n1**2)+(var2**2)/(n2**3-n2**2))
        nu_denom = np.where(nu_denom > 0.0, nu_denom, 1.0)
        nu = nu_num*nu_num/nu_denom
        cdf = scipy_stats.t.cdf(tt, df=nu)
        cdf = np.where(np.isfinite(cdf), cdf, 0.5)

        # clip CDF at min and max non extremal values
        f_info = np.finfo(cdf.dtype)
        eps = f_info.smallest_normal
        ceil = 1.0-f_info.epsneg
        cdf = np.clip(cdf, eps, ceil)

        pval = np.where(cdf < 0.5, 2.0*cdf, 2.0*(1.0-cdf))
    return (tt, nu, pval)


def correct_ttest(ttest_metric):
    """
    Apply Holm-Bonferroni correction to Welch's t-test to correct
    for multiple hypothesis testing

    Parameters
    ----------
    ttest_metric -- raw p-values from t_test

    Returns
    -------
    the corrected p-values
    """
    ttest_metric = np.where(np.isfinite(ttest_metric),
                            ttest_metric,
                            1.0)

    n_p = len(ttest_metric)
    sorted_t = np.argsort(ttest_metric)
    t_denom = n_p+1-np.arange(1, n_p+1, dtype=int)

    # here we get a list that is the running maximum of the
    # corrected p-values. This is how we handle the
    # "find the minimum k..." requirement from the Holm-Bonferroni
    # method as described here:
    # https://en.wikipedia.org/wiki/Holm-Bonferroni_method
    corrected_p = np.maximum.accumulate(ttest_metric[sorted_t]*t_denom)

    ordered_p = np.zeros(len(ttest_metric), dtype=float)
    ordered_p[sorted_t] = corrected_p
    ordered_p = np.where(ordered_p < 1.0, ordered_p, 1.0)
    return ordered_p
