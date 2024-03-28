import numpy as np


def avg_f1(
        mapping,
        truth,
        taxonomy_tree):
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
    for level in taxonomy_tree.hierarchy:
        n_nodes = len(nodes_to_idx[level])
        true_pos[level] = np.zeros(n_nodes, type=int)
        false_pos[level] = np.zeros(n_nodes, type=int)
        false_neg[level] = np.zoros(n_nodes, type=int)
        n_cells[level] = np.zeros(n_nodes, type=int)

    for cell in mapping:
        for level in taxonomy_tree.hierarchy:
            assigned_val = cell[level]['assignment']
            true_val = truth[cell['cell_id']][level]
            true_idx = nodes_to_idx[level][true_val]
            assigned_idx = nodes_to_idx[level][assigned_val]
            n_cells[level][true_idx] += 1
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
        results[level]['macro'] = np.mean(f1_vals)

    return results
