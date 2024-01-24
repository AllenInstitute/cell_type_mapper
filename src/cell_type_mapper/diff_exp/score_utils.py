import numpy as np


def q_score_from_pij(pij_1, pij_2):
    q1_score = np.where(pij_1 > pij_2, pij_1, pij_2)
    denom = np.where(pij_1 > pij_2, pij_1, pij_2)
    denom = np.where(denom > 0.0, denom, 1.0)
    qdiff_score = np.abs(pij_1-pij_2)/denom
    return q1_score, qdiff_score


def pij_from_stats(
        cluster_stats,
        node_1,
        node_2):

    stats_1 = cluster_stats[node_1]
    stats_2 = cluster_stats[node_2]

    pij_1 = stats_1['ge1']/max(1, stats_1['n_cells'])
    pij_2 = stats_2['ge1']/max(1, stats_2['n_cells'])
    log2_fold = np.abs(stats_1['mean']-stats_2['mean'])

    return pij_1, pij_2, log2_fold
