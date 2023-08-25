"""
create precomputed stats file only computing means from the modal
modality (10Xv2, 10Xv3, Multiome) of a given cluster
"""
import h5py
import json
import numpy as np
import pandas as pd
import pathlib

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad)

from cell_type_mapper.taxonomy.taxonomy_tree import (
    TaxonomyTree)

from cell_type_mapper.diff_exp.scores import (
    read_precomputed_stats,
    diffexp_p_values)

def marker_stats(
        cl0,
        cl1,
        stats_path,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        level=None):

    need_to_load = False
    if not hasattr(marker_stats, 'path'):
        need_to_load = True
    elif marker_stats.path != stats_path:
        need_to_load = True

    if need_to_load:
        with h5py.File(stats_path, 'r') as src:
            taxonomy_tree = TaxonomyTree(
                data=json.loads(src['taxonomy_tree'][()].decode('utf-8')))

        taxonomy_tree = taxonomy_tree.drop_level('CCN20230722_SUPT')

        precomputed_stats = read_precomputed_stats(
            precomputed_stats_path=stats_path,
            taxonomy_tree=taxonomy_tree,
            for_marker_selection=True)
        marker_stats.path = stats_path
        marker_stats.tree = taxonomy_tree
        marker_stats.stats = precomputed_stats

    taxonomy_tree = marker_stats.tree
    precomputed_stats = marker_stats.stats
    if level is None:
        level = taxonomy_tree.leaf_level

    pair = (cl0, cl1)

    stats1 = precomputed_stats['cluster_stats'][f'{level}/{pair[0]}']
    stats2 = precomputed_stats['cluster_stats'][f'{level}/{pair[1]}']

    pij1 = stats1['ge1']/max(1, stats1['n_cells'])
    pij2 = stats2['ge1']/max(1, stats2['n_cells'])

    q1 = np.where(pij1>pij2, pij1, pij2)
    denom = np.where(q1>0.0, q1, 1.0)
    qdiff = np.abs(pij1-pij2)/denom

    p_values = diffexp_p_values(
        mean1=stats1['mean'],
        var1=stats1['var'],
        n1=stats1['n_cells'],
        mean2=stats2['mean'],
        var2=stats2['var'],
        n2=stats2['n_cells'])

    return {'p_values': p_values,
            'q1': q1,
            'qdiff': qdiff,
            'n0': stats1['n_cells'],
            'n1': stats2['n_cells']}




