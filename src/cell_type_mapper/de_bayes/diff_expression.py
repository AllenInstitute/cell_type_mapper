from typing import Optional, Tuple, List, Dict, Union, Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
import statsmodels.stats.multitest as multi
import warnings
import logging
import sys

logger = logging.getLogger(__name__)

def vec_chisq_test(pair: tuple,
                  cl_present: pd.DataFrame,
                  cl_size: Dict[Any, int]):
    """
        Vectorized Chi-squared tests for differential gene detection.

        Parameters
        ----------
        pair: a tuple of length 2 specifying which clusters to compare
        cl_present: a data frame of gene detection proportions (clusters x genes)
        cl_size: a dict of cluster sizes

        Returns
        -------
        p_vals: a numpy array of p-values with detection of each gene

    """
    first_cluster = pair[0]
    second_cluster = pair[1]

    cl1_ncells_per_gene = cl_present.loc[first_cluster]*cl_size[first_cluster]
    cl1_ncells = cl_size[first_cluster]
    cl2_ncells_per_gene = cl_present.loc[second_cluster]*cl_size[second_cluster]
    cl2_ncells = cl_size[second_cluster]

    n_genes = cl1_ncells_per_gene.shape[0]

    cl1_present = cl1_ncells_per_gene
    cl1_absent = cl1_ncells - cl1_present
        
    cl2_present = cl2_ncells_per_gene
    cl2_absent = cl2_ncells - cl2_present

    observed = np.array([cl1_present, cl1_absent, cl2_present, cl2_absent])
    p_vals = np.ones(n_genes)
    for i in range(n_genes):
        try:
            chi_squared_stat, p_value, dof, ex = stats.chi2_contingency(observed[:,i].reshape(2,2), correction=True)
            p_vals[i] = p_value
        except:
            logger.debug(f"chi2 exception for cluster pair: {pair}, p value will be assigned to 1")
    return p_vals


def de_pair_chisq(pair: tuple, 
                  cl_present: Union[pd.DataFrame, pd.Series],
                  cl_means: Union[pd.DataFrame, pd.Series],
                  cl_size: Dict[Any, int]) -> pd.DataFrame:
    """
        Perform pairwise differential detection tests using Chi-Squared test for a single pair of clusters.

        Parameters
        ----------
        pair: a tuple of length 2 specifying which clusters to compare
        cl_present: a data frame of gene detection proportions ( clusters x genes) 
        cl_means: a data frame of normalized mean gene expression values ( clusters x genes)
        cl.size: a dict of cluster sizes

        Returns
        -------
        a data frame with differential expressions statistics:
            padj: p-values adjusted
            pval: p-values
            lfc: log fold change of mean expression values between the pair of clusters
            meanA: mean expression value for the first cluster in the pair
            meanB: mean expression value for the second cluster in the pair
            q1: proportion of cells expressing each gene for the first cluster
            q2: proportion of cells expressing each gene for the second cluster
            qdiff: normalized difference between q1 and q2

    """
    if len(pair) != 2:
        raise ValueError("The pair must contain two cluster labels")

    first_cluster, second_cluster = pair

    if isinstance(cl_present, pd.Series):
        cl_present = cl_present.to_frame()
    if isinstance(cl_means, pd.Series):
        cl_means = cl_means.to_frame()

    cl_present_sorted = cl_present[sorted(cl_present.columns)]
    cl_means_sorted = cl_means[sorted(cl_means.columns)]

    if not len(cl_present.columns.difference(cl_means.columns)) == 0:
        raise ValueError("genes names of the cl_means and the cl_present do not match")

    p_vals = vec_chisq_test(pair, 
                            cl_present_sorted,
                            cl_size)
    
    reject, p_adj, alphacSidak, alphacBonf = multi.multipletests(p_vals, method="holm", is_sorted=False)
    lfc = cl_means_sorted.loc[first_cluster].to_numpy() - cl_means_sorted.loc[second_cluster].to_numpy()

    q1 = cl_present_sorted.loc[first_cluster].to_numpy()
    q2 = cl_present_sorted.loc[second_cluster].to_numpy()
    qdiff = get_qdiff(q1, q2)

    de_statistics_chisq = pd.DataFrame(
        {
            "gene": cl_present_sorted.columns.to_list(),
            "p_adj": p_adj,
            "p_value": p_vals,
            "lfc": lfc,
            "meanA": cl_means_sorted.loc[first_cluster].to_numpy(),
            "meanB": cl_means_sorted.loc[second_cluster].to_numpy(),
            "q1": q1,
            "q2": q2,
            "qdiff": qdiff,
        }
    )
    
    de_statistics_chisq.set_index("gene")

    return de_statistics_chisq


def filter_gene_stats(
    de_stats: pd.DataFrame,
    gene_type: str,
    cl1_size: float = None,
    cl2_size: float = None,
    q1_thresh: float = None,
    q2_thresh: float = None,
    cluster_size_thresh: int = None,
    qdiff_thresh: float = None,
    padj_thresh: float = None,
    lfc_thresh: float = None

) -> pd.DataFrame:
    """
    Filter out differential expression summary stats
    for either up-regulated or down-regulated genes

    Parameters
    ----------
    de_stats:
        dataframe with stats for each gene
    gene_type:
    cl1_size:
        cluster size of the first cluster
    cl2_size:
        cluster size of the second cluster
    q1_thresh:
        threshold for proportion of cells expressing each gene in the first cluster
    q2_thresh:
        threshold for proportion of cells expressing each gene in the second cluster
    cluster_size_thresh:
        threshold for min number of cells in cluster
    qdiff_thresh:
        threshold for qdiff
    padj_thresh:
        threshold for padj
    lfc_thresh:
        threshold for lfc

    Returns
    -------
    filtered dataframe
    """
    if gene_type == 'up-regulated':
        mask = de_stats['lfc'] > 0
        qa = 'q1'
        qb = 'q2'
        cl_size = cl1_size
    elif gene_type == 'down-regulated':
        mask = de_stats['lfc'] < 0
        qa = 'q2'
        qb = 'q1'
        cl_size = cl2_size
    else:
        raise ValueError(f"Invalid gene_type value {gene_type}. "
                         f"Allowed values include 'up-regulated' and 'down-regulated'. ")

    if padj_thresh:
        mask &= de_stats['p_adj'] < padj_thresh
    if lfc_thresh:
        mask &= abs(de_stats['lfc']) > lfc_thresh

    if q1_thresh:
        mask &= de_stats[qa] > q1_thresh
    if cl_size:
        mask &= de_stats[qa] * cl_size >= cluster_size_thresh
    if q2_thresh:
        mask &= de_stats[qb] < q2_thresh
    if qdiff_thresh:
        mask &= abs(de_stats['qdiff']) > qdiff_thresh

    return de_stats.loc[mask]


def get_qdiff(q1, q2) -> np.array:
    """
    Calculate normalized difference between q1 and q2 proportions
    when q1=0 and q2=0, return qdiff=0
    Parameters
    ----------
    q1:
        proportion of cells expressing each gene  for the first cluster
    q2:
        proportion of cells expressing each gene  for the second cluster

    Returns
    -------
    qdiff statistic
    """
    qmax = np.maximum(q1, q2)
    qmax[qmax == 0] = np.nan
    q_diff = abs(q1 - q2) / qmax
    np.nan_to_num(q_diff, nan=0.0, copy=False)

    return q_diff


def calc_de_score(
    padj: np.ndarray
) -> float:
    """
    Calculate DE score for a group of cells from padj values

    Parameters
    ----------
    padj:
        adjusted p-value

    Returns
    -------
    differential expression score
    """
    with np.errstate(divide='ignore'): # allow de_score = inf for padj = 0
        de_score = np.sum(-np.log10(padj)) if len(padj) else 0

    return de_score

def de_pairs_chisq(
        pairs: List[Tuple[Any, Any]],
        cl_means: pd.DataFrame,
        cl_present: pd.DataFrame,
        cl_size: Dict[Any, int],
        de_thresholds: Dict[str, Any],
    ):
    """
    Compute Chi square test for pairs of cluster

    Parameters
    ----------
    pairs: list of pairs of cluster names
    cl_means: a data frame of normalized mean gene expression values ( clusters x genes)
    cl_present: a data frame of gene detection proportions ( clusters x genes)
    cl_size: dict of cluster name: number of observations in cluster
    de_thresholds: thresholds for filter de

    Returns
    -------
    Dict with key: cluster_pair, value: dict of de values
    """
    de_pairs = {}
    for pair in pairs:
        # chisq test
        de_pair_stats = de_pair_chisq(pair, cl_present, cl_means, cl_size)

        # get de score
        first_cluster, second_cluster = pair

        de_pair_up = filter_gene_stats(
            de_stats=de_pair_stats,
            gene_type='up-regulated',
            cl1_size=cl_size[first_cluster],
            cl2_size=cl_size[second_cluster],
            **de_thresholds
        )
        up_score = calc_de_score(de_pair_up['p_adj'].values)

        de_pair_down = filter_gene_stats(
            de_stats=de_pair_stats,
            gene_type='down-regulated',
            cl1_size=cl_size[first_cluster],
            cl2_size=cl_size[second_cluster],
            **de_thresholds
        )
        down_score = calc_de_score(de_pair_down['p_adj'].values)

        de_pairs[pair] = {
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
