import numpy as np


def avg_f1(
        mapping,
        truth,
        taxonomy_tree,
        probability_cut_list=None,
        correlation_cut_list=None):
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
    probability_cut_list:
        List of values on which to cut in probability
        (i.e. if aggregate probability is lower than this
        threshold, assume the label is false)
    correlation_cut_list:
        List of values on which to cut in avg. correlation
        coefficient.

    Returns
    -------
    A dict that maps
       level -> "probability/correlation" -> cut -> stats

    stats is in the form of a dict:
        micro -- micro-averaged F1
        macro -- macro-averaged F1
        macro_adjusted -- macro-averaged F1 with NaNs converted to 0
        true_pos -- number of true positives
        true_neg -- 0; unclear how to handle this
        false_pos -- number of false positives
        false_neg -- number of false negatives
        n_cells -- number of cells (uninteresting)
        valid_classes -- number of cell types at this level with
                         finite F1
        est_false_pos -- number of false positives you would estimate
                         naively believing aggregate probability
                         (0 for cuts in average correlation)
    """
    if probability_cut_list is None:
        probability_cut_list = [0.0]
    if correlation_cut_list is None:
        correlation_cut_list = []

    cut_list = [
        ('probability', v) for v in probability_cut_list
    ]
    cut_list += [
        ('correlation', v) for v in correlation_cut_list
    ]


    nodes_to_idx = dict()
    for level in taxonomy_tree.hierarchy:
        nodes_to_idx[level] = dict()
        for ii, node in enumerate(taxonomy_tree.nodes_at_level(level)):
            nodes_to_idx[level][node] = ii

    true_pos = dict()
    false_pos = dict()
    false_neg = dict()
    true_neg = dict()
    n_cells = dict()
    n_tot = dict()
    estimated_false_pos = dict()
    for level in taxonomy_tree.hierarchy:
        n_nodes = len(nodes_to_idx[level])
        n_cells[level] = np.zeros(n_nodes, dtype=int)
        n_tot[level] = 0

        for lookup in (true_pos,
                       false_pos,
                       false_neg,
                       true_neg,
                       estimated_false_pos):
            lookup[level] = dict()

        for cut in cut_list:
            true_pos[level][cut] = np.zeros(n_nodes, dtype=int)
            false_pos[level][cut] = np.zeros(n_nodes, dtype=int)
            false_neg[level][cut] = np.zeros(n_nodes, dtype=int)
            true_neg[level][cut] = np.zeros(n_nodes, dtype=int)
            estimated_false_pos[level][cut] = 0

    for cell in mapping:
        agg_prob = 1.0
        for level in taxonomy_tree.hierarchy:
            assigned_val = cell[level]['assignment']
            agg_prob *= cell[level]['bootstrapping_probability']
            true_val = truth[cell['cell_id']][level]
            true_idx = nodes_to_idx[level][true_val]
            n_cells[level][true_idx] += 1
            n_tot[level] += 1

            assigned_idx = nodes_to_idx[level][assigned_val]
            if assigned_idx == true_idx:
                is_true = True
            else:
                is_true = False

            for cut in cut_list:
                considered_true = True
                if cut[0] == 'probability':
                     if agg_prob < cut[1]:
                         considered_true=False
                elif cut[0] == 'correlation':
                    if cell[level]['avg_correlation'] < cut[1]:
                        considered_true=False
                else:
                    raise RuntimeError(
                        f"Do not know how to handle cut {cut}"
                    )

                if considered_true:
                    if cut[0] == 'probability':
                        estimated_false_pos[level][cut] += (1.0-agg_prob)

                if is_true:
                    if considered_true:
                        true_pos[level][cut][true_idx] += 1
                    else:
                        false_neg[level][cut][true_idx] += 1
                else:
                    false_neg[level][cut][true_idx] += 1
                    if considered_true:
                        false_pos[level][cut][assigned_idx] += 1

    results = dict()
    for level in taxonomy_tree.hierarchy:
        results[level] = dict()
        for cut in cut_list:
            if cut[0] not in results[level]:
                results[level][cut[0]] = dict()

            this = dict()

            tp = true_pos[level][cut]
            fp = false_pos[level][cut]
            fn = false_neg[level][cut]
            tp_sum = tp.sum()
            this['micro'] = tp_sum/(tp_sum+0.5*(fn.sum() + fp.sum()))
            f1_vals = tp/(tp+0.5*(fn+fp))
            this['macro'] = np.nanmean(f1_vals)
            adj = np.where(np.isfinite(f1_vals), f1_vals, 0.0)
            this['macro_adjusted'] = np.mean(adj)
            this['true_pos'] = int(true_pos[level][cut].sum())
            this['true_neg'] = int(true_neg[level][cut].sum())
            this['false_pos'] = int(false_pos[level][cut].sum())
            this['false_neg'] = int(false_neg[level][cut].sum())
            this['n_cells'] = n_tot[level]
            this['valid_classes'] = int(np.isfinite(f1_vals).sum())
            if cut[0] == 'probability':
                this['est_false_pos'] = estimated_false_pos[level][cut]
            results[level][cut[0]][f'{cut[1]:.2f}'] = this

    return results
