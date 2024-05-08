import numpy as np


def avg_f1(
        mapping,
        truth,
        taxonomy_tree,
        probability_cut=None):
    """
    Return the averaged f1 scores for a mapping (both micro and macro)

    Parameters
    ----------
    mapping:
        List of dicts representing the mapping of the cells
        (this is the 'results' entry in the JSON output produced
        by the mapper)
    truth:
        A dict mapping each cell_id to its true mapping, so
        {cell1: {
            level1: truth1,
            level2: truth2,
            ...},
         cell2:...
        }
    taxonomy_tree:
        The TaxonomyTree associated with the mapping
    probability_cut:
        If not None, ignore cells with
        aggregate_probability < probability_cut
        (if aggregate_probability available)

    Returns
    -------
    A dict
       {level1: {'micro': micro_avg_f1, 'macro': macro_avg_f1},
        level2: {'micro': ....}
       }
    """
    nodes_to_idx = dict()
    for level in taxonomy_tree.hierarchy:
        nodes_to_idx[level] = dict()
        for ii, node in enumerate(taxonomy_tree.nodes_at_level(level)):
            nodes_to_idx[level][node] = ii

    true_pos = dict()
    false_pos = dict()
    false_neg = dict()
    n_cells = dict()
    n_tot = dict()
    for level in taxonomy_tree.hierarchy:
        n_nodes = len(nodes_to_idx[level])
        true_pos[level] = np.zeros(n_nodes, dtype=int)
        false_pos[level] = np.zeros(n_nodes, dtype=int)
        false_neg[level] = np.zeros(n_nodes, dtype=int)
        n_cells[level] = np.zeros(n_nodes, dtype=int)
        n_tot[level] = 0

    n_skipped = 0
    for cell in mapping:
        for level in taxonomy_tree.hierarchy:
            if probability_cut is not None:
                if 'aggregate_probability' in cell[level]:
                    actual = cell[level]['aggregate_probability']
                    if actual < probability_cut:
                        n_skipped += 1
                        continue

            assigned_val = cell[level]['assignment']
            true_val = truth[cell['cell_id']][level]
            true_idx = nodes_to_idx[level][true_val]
            assigned_idx = nodes_to_idx[level][assigned_val]
            n_cells[level][true_idx] += 1
            n_tot[level] += 1
            if assigned_val == true_val:
                true_pos[level][true_idx] += 1
            else:
                false_neg[level][true_idx] += 1
                false_pos[level][assigned_idx] += 1

    results = dict()
    for level in taxonomy_tree.hierarchy:
        results[level] = dict()
        tp = true_pos[level]
        fp = false_pos[level]
        fn = false_neg[level]
        tp_sum = tp.sum()
        results[level]['micro'] = tp_sum/(tp_sum+0.5*(fn.sum() + fp.sum()))
        f1_vals = tp/(tp+0.5*(fn+fp))
        results[level]['macro'] = np.nanmean(f1_vals)
        adj = np.where(np.isfinite(f1_vals), f1_vals, 0.0)
        results[level]['macro_adjusted'] = np.mean(adj)
        results[level]['n_cells'] = n_tot[level]
        results[level]['valid_classes'] = int(np.isfinite(f1_vals).sum())

    return results
