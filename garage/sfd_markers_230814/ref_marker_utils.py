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

from cell_type_mapper.de_bayes.de_ebayes import (
    de_pairs_ebayes)

def marker_stats(
        cl0,
        cl1,
        stats_path,
        p_th=0.01,
        q1_th=0.5,
        qdiff_th=0.7,
        level=None):
    need_to_load = False
    if not hasattr(marker_stats, 'lookup'):
        need_to_load = True
        marker_stats.lookup = dict()
    elif stats_path not in marker_stats.lookup:
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
        marker_stats.tree = taxonomy_tree
        marker_stats.lookup[stats_path] = precomputed_stats

    precomputed_stats = marker_stats.lookup[stats_path]
    taxonomy_tree = marker_stats.tree

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

    gene_names = [f'g_{ii}' for ii in range(len(stats1['mean']))]
    pairs = [(cl0, cl1)]

    cl_mean_data = []
    cl_var_data = []
    for cl, stats in zip((cl0, cl1), (stats1, stats2)):
        this_mean = {'cluster_name': cl}
        this_var = {'cluster_name': cl}
        for ii, gene in enumerate(gene_names):
            this_mean[gene] = stats['mean'][ii]
            this_var[gene] = stats['var'][ii]
        cl_mean_data.append(this_mean)
        cl_var_data.append(this_var)
    cl_means = pd.DataFrame(cl_mean_data).set_index('cluster_name')
    cl_vars = pd.DataFrame(cl_var_data).set_index('cluster_name')
    cl_size = {cl0: stats1['n_cells'], cl1: stats2['n_cells']}

    de_p_vals = de_pairs_ebayes(
        pairs=pairs,
        cl_means=cl_means,
        cl_vars=cl_vars,
        cl_size=cl_size,
        p_th=p_th)

    #de_p_vals= {pairs[0]: None}

    return {'p_values': p_values,
            'de_p_values': de_p_vals[pairs[0]],
            'q1': q1,
            'qdiff': qdiff,
            'n0': stats1['n_cells'],
            'n1': stats2['n_cells']}




