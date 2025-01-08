import numpy as np


def mapping_cdf(
        mapping,
        truth,
        taxonomy_tree,
        bin_resolution=0.01):
    """
    Return the actual and expected cumulative probability
    distribution functions of accuracy as a function of
    aggregated bootstrapping probability.

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
    bin_resolution:
        Width of probability bins in which to subdivide the
        CDF.

    Returns
    -------
    A dict
       {level1: {'bins': array of aggregate_probability bins,
                 'actual': actual CDF of "is the assignment correct",
                 'expected': expected CDF based on aggregate_probability
                }
        level2: {...}
        .....
       }
    """
    bins = np.arange(bin_resolution, 1.0+bin_resolution, bin_resolution)

    true_prob_lookup = dict()
    false_prob_lookup = dict()
    conditioned_true_prob_lookup = dict()
    conditioned_false_prob_lookup = dict()
    for level in taxonomy_tree.hierarchy:
        true_prob_lookup[level] = []
        false_prob_lookup[level] = []
        conditioned_true_prob_lookup[level] = []
        conditioned_false_prob_lookup[level] = []

    for i_cell, cell in enumerate(mapping):
        for i_level, level in enumerate(taxonomy_tree.hierarchy):

            conditioned = False
            if i_level == 0:
                conditioned = True
            else:
                parent = taxonomy_tree.hierarchy[i_level-1]
                cell_parent = cell[parent]['assignment']
                true_parent = truth[cell['cell_id']][parent]
                if cell_parent == true_parent:
                    conditioned = True

            prob = cell[level]['aggregate_probability']
            level_prob = cell[level]['bootstrapping_probability']

            if cell[level]['assignment'] == truth[cell['cell_id']][level]:
                true_prob_lookup[level].append(prob)
                if conditioned:
                    conditioned_true_prob_lookup[level].append(level_prob)
            else:
                false_prob_lookup[level].append(prob)
                if conditioned:
                    conditioned_false_prob_lookup[level].append(level_prob)

    result = dict()
    for level in taxonomy_tree.hierarchy:
        true_prob = np.array(true_prob_lookup[level])
        false_prob = np.array(false_prob_lookup[level])

        (actual,
         expected) = _return_cdf(
                         true_prob=true_prob,
                         false_prob=false_prob,
                         bins=bins)

        c_true = np.array(conditioned_true_prob_lookup[level])
        c_false = np.array(conditioned_false_prob_lookup[level])
        (c_actual,
         c_expected) = _return_cdf(
                         true_prob=c_true,
                         false_prob=c_false,
                         bins=bins)

        result[level] = {
            'bins': bins,
            'actual': actual,
            'expected': expected,
            'conditioned_actual': c_actual,
            'conditioned_expected': c_expected
        }

    return result


def area_between_cdf(
        mapping,
        truth,
        taxonomy_tree,
        bin_resolution=0.01):
    """
    Return the area between the actual and expected CDF of
    cell types as a function of aggregated bootstrapping
    probability.

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
    bin_resolution:
        Width of probability bins in which to subdivide the
        CDF.

    Returns
    -------
    A dict
       {level1: {'area': area,
                 'signed_area': positive is over confident; negative is under
                                confident
                }
        level2: {...}
        .....
       }

    There will also be 'conditioned_area' and 'conditioned_signed_area'
    which are the same metrics, but only considering those cells who
    were correctly assigned at the parent level of the given level.
    """
    cdf_lookup = mapping_cdf(
        mapping=mapping,
        truth=truth,
        taxonomy_tree=taxonomy_tree,
        bin_resolution=bin_resolution
    )

    result = dict()
    for level in taxonomy_tree.hierarchy:
        cdf = cdf_lookup[level]

        area = _riemann_area(
            x=cdf['binned'],
            true_y=cdf['expected'],
            actual_y=cdf['actual'],
            signed=False)

        signed_area = _riemann_area(
            x=cdf['bins'],
            true_y=cdf['expected'],
            actual_y=cdf['actual'],
            signed=True)

        result[level] = {
            'area': area,
            'signed_area': signed_area
        }

        c_area = _riemann_area(
            x=cdf['bins'],
            true_y=cdf['conditioned_expected'],
            actual_y=cdf['conditioned_actual'],
            signed=False)

        c_signed_area = _riemann_area(
            x=cdf['bins'],
            true_y=cdf['conditioned_expected'],
            actual_y=cdf['conditioned_actual'],
            signed=True)

        result[level]['conditioned_area'] = c_area
        result[level]['conditioned_signed_area'] = c_signed_area

    return result


def _return_cdf(
        true_prob,
        false_prob,
        bins):
    all_prob = np.concatenate([true_prob, false_prob])
    expected = np.zeros(len(bins), dtype=float)
    actual = np.zeros(len(bins), dtype=float)
    for ii, bb in enumerate(bins):
        all_mask = (all_prob >= bb)

        if len(all_mask) > 0:
            expected[ii] = all_prob[all_mask].sum()/max(1, all_mask.sum())

        true_mask = (true_prob >= bb)
        if len(true_mask) > 0:
            this_true = true_mask.sum()
        else:
            this_true = 0.0

        false_mask = (false_prob >= bb)
        if len(false_mask) > 0:
            this_false = false_mask.sum()
        else:
            this_false = 0.0

        actual[ii] = this_true/(max(1, this_true+this_false))
    return actual, expected


def _riemann_area(
        x,
        true_y,
        actual_y,
        signed=False):

    dy = true_y-actual_y
    dx = x[1:]-x[:-1]
    if signed:
        area_arr = dx*0.5*(dy[1:]+dy[:-1])
    else:
        area_arr = np.abs(dx*0.5*(dy[1:]+dy[:-1]))

    return area_arr.sum()
