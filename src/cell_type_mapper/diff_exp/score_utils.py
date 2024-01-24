import numpy as np


def q_score_from_pij(pij_1, pij_2):
    q1_score = np.where(pij_1 > pij_2, pij_1, pij_2)
    denom = np.where(pij_1 > pij_2, pij_1, pij_2)
    denom = np.where(denom > 0.0, denom, 1.0)
    qdiff_score = np.abs(pij_1-pij_2)/denom
    return q1_score, qdiff_score
