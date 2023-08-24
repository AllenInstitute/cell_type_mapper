from typing import Optional, Tuple, List, Dict, Union, Any
from numpy.core.fromnumeric import var
from numpy.typing import ArrayLike

import warnings
import logging
import time

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.special import digamma, polygamma
from statsmodels.stats.multitest import multipletests

from .diff_expression import get_qdiff, filter_gene_stats, calc_de_score

logger = logging.getLogger(__name__)

"""
Implements functions for calculating differential expression
through moderated t-statistics as defined in 
Smyth, 2004 and Phipson et al., 2016

This module greatly simplifies the full process of
- Create dummy coding design with each cluster as an experimental condition (no intercept)
- fit linear model for each gene, calculated coefficients, residuals, degrees of freedom, etc
- moderate gene expression residual variances using empirical bayes
- create a contrast and update fit for cluster pair of interest
- perform t-test on each contrast fit
by recognizing 
- coefficients are means of each cluster,
- variances and degrees of freedom can be calculated directly from mean and mean squared values

"""

def trigamma_inverse(x, tol=1e-08, iter_limit=50):
    """Newton's method to solve trigamma inverse"""
    y = 0.5 + 1 / x
    for i in range(iter_limit):
        tri = polygamma(1, y)
        diff = tri * (1 - tri / x) / polygamma(2, y)
        y += diff
        if np.max(-diff / y) < tol:
            break
    else:
        warnings.warn(
            "trigamma_inverse iteration limit ({iter_limit}) exceeded"
        )
    return y


def fit_f_dist(x: ArrayLike, df1: ArrayLike):
    """
    Method of moments to fit f-distribution
    
    Parameters
    ----------
    x: samples
    df1: degrees of freedom
    
    Returns
    -------
    df2, scale parameters for f-distribution
    """
    z = np.log(x)
    e = z - digamma(df1 / 2) + np.log(df1 / 2)

    e_mean = np.mean(e)
    e_var = np.var(e, ddof=1)

    e_var -= np.mean(polygamma(1, df1 / 2))

    if e_var > 0:
        df2 = 2 * trigamma_inverse(e_var)
        scale = np.exp(e_mean + digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        scale = np.exp(e_mean)

    return df2, scale


def moderate_variances(
        variances: pd.DataFrame,
        df: int,
    ):
    """
    Moderated variances 
    
    - Assume each gene's variance is sampled from a 
      scaled inverse chi-square prior distribution
      with degrees of freedom d0 and location s_0^2 (sigma_0 squared)
    - Fit fDist to get prior variance
    - Get posterior variance from sample variance and prior variance 
    
    
    Parameters
    ----------
    variances: sample variances (index = gene)
    df: degrees of freedom
    
    Returns
    -------
    pd.DataFrame moderated (posterior) variances
    float prior variance
    float prior degree of freedom
    """

    var = np.squeeze(variances.to_numpy())
    idxs_zero = np.where(var == 0)[0]
    if idxs_zero.size > 0:
        logger.debug(f'offsetting zero variances from zero')
        var[idxs_zero] += np.finfo(var.dtype).eps

    df_prior, var_prior = fit_f_dist(var, df)
    var_post = (df_prior * var_prior + df * var) / (df + df_prior)

    var_post = pd.DataFrame(var_post, index=variances.index)

    return var_post, var_prior, df_prior


def get_linear_fit_vals(cl_vars: pd.DataFrame, cl_size: Dict[Any, int]):
    """
    Directly computes sigma squared, degrees of freedom, and stdev_unscaled
    for a linear fit of clusters from cluster variances and cluster size
    """
    cl_size_v = np.asarray([cl_size[clust] for clust in cl_vars.index])
    df = cl_size_v.sum() - len(cl_size_v)
    sigma_sq = cl_vars.T @ cl_size_v / df

    stdev_unscaled = pd.DataFrame(1 / np.sqrt(cl_size_v), index=cl_vars.index)
    return sigma_sq.to_frame(), df, stdev_unscaled


def de_pairs_ebayes(
        pairs: List[Tuple[Any, Any]],
        cl_means: pd.DataFrame,
        cl_vars: pd.DataFrame,
        cl_present: pd.DataFrame,
        cl_size: Dict[Any, int],
        de_thresholds: Dict[str, Any],
    ):
    """
    Computes moderated t-statistics for pairs of cluster

    Steps:
        Get sigma squared and degrees of freedom as if a linear model was fit for each gene
        Moderate all gene variances (see moderate variances for details)
        Compute cluster pair t-test p-val for all genes using moderated variances
        Adjust cluster pair pvals
        Filter and compute descore for pair
    
    Parameters
    ----------
    pairs: list of pairs of cluster names
    cl_means: dataframe with index = cluster name, columns = genes,
              values = per cluster mean of gene expression (E[X])
    cl_vars: dataframe with index = cluster name, columns = genes,
                 values = per cluster variance of gene expression
    cl_size: dict of cluster name: number of observations in cluster
    de_thresholds: thresholds for filter de

    Returns
    -------
    Dict with key: cluster_pair, value: dict of de values
    """
    logger.info('Fitting Variances')
    sigma_sq, df, stdev_unscaled = get_linear_fit_vals(cl_vars, cl_size)
    logger.info('Moderating Variances')
    sigma_sq_post, var_prior, df_prior = moderate_variances(sigma_sq, df)

    logger.info(f'Comparing {len(pairs)} pairs')
    de_pairs = {}
    for (cluster_a, cluster_b) in pairs:
        # t-test with ebayes adjusted variances
        means_diff = cl_means.loc[cluster_a] - cl_means.loc[cluster_b]
        means_diff = means_diff.to_frame()
        stdev_unscaled_comb = np.sqrt(np.sum(stdev_unscaled.loc[[cluster_a, cluster_b]] ** 2))[0]
        
        df_total = df + df_prior
        df_pooled = np.sum(df)
        df_total = min(df_total, df_pooled)
        
        t_vals = means_diff / np.sqrt(sigma_sq_post) / stdev_unscaled_comb
        
        p_adj = np.ones((len(t_vals),))
        p_vals = 2 * stats.t.sf(np.abs(t_vals[0]), df_total)
        reject, p_adj, alphacSidak, alphacBonf= multipletests(p_vals, alpha=de_thresholds['padj_thresh'], method='holm')
        lfc = means_diff

        # Get DE score
        de_pair_stats = pd.DataFrame(index=cl_means.columns)
        de_pair_stats['p_value'] = p_vals
        de_pair_stats['p_adj'] = p_adj
        de_pair_stats['lfc'] = lfc
        de_pair_stats["meanA"] = cl_means.loc[cluster_a]
        de_pair_stats["meanB"] = cl_means.loc[cluster_b]
        de_pair_stats["q1"] = cl_present.loc[cluster_a]
        de_pair_stats["q2"] = cl_present.loc[cluster_b]
        de_pair_stats["qdiff"] = get_qdiff(cl_present.loc[cluster_a], cl_present.loc[cluster_b])

        de_pair_up = filter_gene_stats(
            de_stats=de_pair_stats,
            gene_type='up-regulated', 
            cl1_size=cl_size[cluster_a],
            cl2_size=cl_size[cluster_b],
            **de_thresholds
        )
        up_score = calc_de_score(de_pair_up['p_adj'].values)

        de_pair_down = filter_gene_stats(
            de_stats=de_pair_stats,
            gene_type='down-regulated',
            cl1_size=cl_size[cluster_a],
            cl2_size=cl_size[cluster_b],
            **de_thresholds
        )
        down_score = calc_de_score(de_pair_down['p_adj'].values)

        de_pairs[(cluster_a, cluster_b)] = {
            'score': up_score + down_score,
            'up_score': up_score,
            'down_score': down_score,
            'up_genes': de_pair_up.index.to_list(),
            'down_genes': de_pair_down.index.to_list(),
            'up_num': len(de_pair_up.index),
            'down_num': len(de_pair_down.index),
            'num': len(de_pair_up.index) + len(de_pair_down.index)
        }

    de_pairs = pd.DataFrame(de_pairs).T
    return de_pairs
