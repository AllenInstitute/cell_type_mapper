import h5py
import json
import numpy as np
import warnings

from cell_type_mapper.utils.utils import (
    choose_int_dtype)


def q_score_from_pij(pij_1, pij_2):
    """
    Return the q1 and qdiff scores for each gene comparing
    two clusters (q1 and qdiff are defined in Tasic et al. 2018).

    Parameters
    ----------
    pij_1:
        A (n_genes,) numpy array containing the fraction of each
        cell in cluster_1 that expresses each gene at >= 1CPM
    pij_2:
        Same as pij_1 for cluster_2

    Returns
    -------
    q1_score:
        A (n_genes,) numpy array expressing, for each gene,
        the greate r of pij_1 or pij_2
    qdiff_score:
        A (n_genes,) numpy array expressing, for each gene
        |pij_1-pij_2|/max(pij_1, pij_2)
    """
    q1_score = np.where(pij_1 > pij_2, pij_1, pij_2)
    denom = np.where(pij_1 > pij_2, pij_1, pij_2)
    denom = np.where(denom > 0.0, denom, 1.0)
    qdiff_score = np.abs(pij_1-pij_2)/denom
    return q1_score, qdiff_score


def pij_from_stats(
        cluster_stats,
        node_1,
        node_2):
    """
    For two cell type clusters return:
        The fraction of each cell in cluster_1 that expresses each
        gene at >= 1 CPM (pij_2)

        The fraction of each cellin cluster_2 that expresses each
        gene at >= 1 CPM (pij_2)

        The absolute value of the difference in the mean log2(CPM+1)
        expression of each gene between the two clusters (log2_fold)

    Parameters
    ----------
    cluster_stats:
        A dict mapping cluster names to lookup tables of stats
        about each gene in that cluster. Must at least map to
        'n_cells', 'mean', and 'ge1'
        (see read_precomputed_stats)

    node_1:
        Key to cluster stats denoting cluster_1

    node_2:
        Key to cluster stats denoting cluster_2

    Returns
    -------
    pij_1
    pij_2
    log2_fold
    (see above)
    """
    stats_1 = cluster_stats[node_1]
    stats_2 = cluster_stats[node_2]

    pij_1 = stats_1['ge1']/max(1, stats_1['n_cells'])
    pij_2 = stats_2['ge1']/max(1, stats_2['n_cells'])
    log2_fold = np.abs(stats_1['mean']-stats_2['mean'])

    return pij_1, pij_2, log2_fold


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
    this_cluster_stats = dict()
    this_tree_as_leaves = dict()
    for idx in idx_to_pair:
        sibling_pair = idx_to_pair[idx]
        level = sibling_pair[0]
        if level not in this_tree_as_leaves:
            this_tree_as_leaves[level] = dict()
        for node in (sibling_pair[1], sibling_pair[2]):
            node_k = f'{level}/{node}'
            this_cluster_stats[node_k] = cluster_stats[node_k]
            if node not in this_tree_as_leaves[level]:
                this_tree_as_leaves[level][node] = tree_as_leaves[level][node]

    return this_cluster_stats, this_tree_as_leaves


def read_precomputed_stats(
        precomputed_stats_path,
        taxonomy_tree,
        for_marker_selection=True):
    """
    Read precomputed stats from precomputed stats path.

    Return a dict
    {'gene_names': [list, of, gene, names],
     'cluster_stats': {
         Dict mapping 'level/node_name' to 'mean', 'var', 'ge1'
     }
    }

    If for_marker_selection = False, do not complain if you cannot
    compute 'var', 'ge1' (which are only needed if selecting marker
    genes)
    """
    raw_results = read_raw_precomputed_stats(
            precomputed_stats_path=precomputed_stats_path,
            for_marker_selection=for_marker_selection)

    results = dict()
    results['gene_names'] = raw_results['gene_names']
    results['cluster_stats'] = dict()
    as_leaves = taxonomy_tree.as_leaves
    for level in as_leaves:
        for node in as_leaves[level]:
            leaf_population = as_leaves[level][node]
            this = aggregate_stats(
                leaf_population=leaf_population,
                precomputed_stats=raw_results['cluster_stats'])

            key_list = list(this.keys())
            for key in key_list:
                if key not in ('mean', 'var', 'ge1', 'n_cells'):
                    this.pop(key)
            if not for_marker_selection:
                for key in ('var', 'ge1', 'n_cells'):
                    if key in this:
                        this.pop(key)
            results['cluster_stats'][f'{level}/{node}'] = this
    return results


def read_raw_precomputed_stats(
        precomputed_stats_path,
        for_marker_selection=True):
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

    if for_marker_selection is True and 'sumsq' or 'ge1' are missing,
    raise an error
    """

    precomputed_stats = dict()
    raw_data = dict()
    with h5py.File(precomputed_stats_path, 'r') as in_file:

        precomputed_stats['gene_names'] = json.loads(
            in_file['col_names'][()].decode('utf-8'))

        row_lookup = json.loads(
            in_file['cluster_to_row'][()].decode('utf-8'))

        # cast indexes to integers
        row_lookup = {k: int(row_lookup[k]) for k in row_lookup}

        all_keys = set(['n_cells', 'sum', 'sumsq', 'gt0', 'gt1', 'ge1'])
        all_keys = list(all_keys.intersection(set(in_file.keys())))

        if 'n_cells' not in all_keys or 'sum' not in all_keys:
            raise RuntimeError(
                "'n_cells' and 'sum' must be in precomputed stats "
                f"file. The file\n{precomputed_stats_path}\n"
                f"contains {in_file.keys()}")

        if for_marker_selection:
            if 'sumsq' not in all_keys or 'ge1' not in all_keys:
                raise RuntimeError(
                    "'sumsq' and 'ge1' must be in precomputed stats "
                    "file in order to use it for marker selection. The "
                    f"file\n{precomputed_stats_path}\n"
                    f"contains {in_file.keys()}")

        for k in all_keys:
            if k in in_file:
                raw_data[k] = in_file[k][()]

    cluster_stats = dict()
    for leaf_name in row_lookup:
        idx = row_lookup[leaf_name]
        this = dict()
        if 'n_cells' in raw_data:
            this['n_cells'] = raw_data['n_cells'][idx]
        for k in all_keys:
            if k == 'n_cells':
                continue
            if k in raw_data:
                this[k] = raw_data[k][idx, :]
        cluster_stats[leaf_name] = this

    precomputed_stats['cluster_stats'] = cluster_stats
    return precomputed_stats


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

        if 'n_cells' in these_stats:
            n_cells += these_stats['n_cells']

        if 'sum' in these_stats:
            sum_arr += these_stats['sum']

        if 'sumsq' in these_stats:
            sumsq_arr += these_stats['sumsq']

        if 'gt0' in these_stats:
            gt0 += these_stats['gt0'].astype(int)

        if 'gt1' in these_stats:
            gt1 += these_stats['gt1'].astype(int)

        if 'ge1' in these_stats:
            ge1 += these_stats['ge1'].astype(int)
        else:
            has_ge1 = False

    mu = sum_arr/max(1, n_cells)
    var = (sumsq_arr-sum_arr**2/max(1, n_cells))/max(1, n_cells-1)

    if not has_ge1:
        warnings.warn("precomputed stats file does not have 'ge1' data")
        ge1 = None

    result = {'mean': mu,
              'var': var,
              'n_cells': n_cells,
              'gt0': gt0,
              'gt1': gt1,
              'ge1': ge1}

    for k in ('gt0', 'gt1', 'ge1'):
        if result[k] is not None:
            new_dtype = choose_int_dtype(
                (result[k].min(), result[k].max()))
            result[k] = result[k].astype(new_dtype)

    return result
