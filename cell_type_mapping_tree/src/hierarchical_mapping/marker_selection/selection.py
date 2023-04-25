import numpy as np
import time

from hierarchical_mapping.utils.taxonomy_utils import (
    get_all_leaf_pairs)

from hierarchical_mapping.corr.utils import (
    match_genes)

from hierarchical_mapping.marker_selection.utils import (
    create_utility_array)


def select_marker_genes_v2(
        marker_gene_array,
        query_gene_names,
        taxonomy_tree,
        parent_node,
        n_per_utility=15):
    """
    Select marker genes given a reference set and a query set.

    Each gene either is or is not a marker (no ranking by score).

    Genes will be selected based on how many taxonomy pairs they
    can distinguish between.

    Try to find an equal number of up- and down- regulated markers
    for each taxonomy pair (i.e. a balance of genes that are more
    prominent in pair[0] versus more prominent in pair[1])

    Parameters
    ----------
    marker_gene_array:
        MarkerGeneArray providing access to data computed by

        diff_exp.markers.find_markers_for_all_taxonomy_pairs.

        This lists data about which genes are marker genes in
        the reference dataset without regards to any query set.

    query_gene_names:
        List of gene names in a query set

    taxonomy_tree:
        dict encoding the taxonomy we are mapping to

    parent_node:
        (level, node) tuple indicating the parent for whose
        children we are selecting markers (None will indicate
        that we are at the root of the tree)

    n_per_utility:
        Number of marker genes to select per (node1, node2, +/-) set
        (+/- indicates which node the gene is more prominent in)

    Returns
    -------
    A list of marker gene names.
        (Alphabetized for lack of a better ordering scheme.)
    """
    t0 = time.time()

    # get a numpy array of indices indicating which taxonomy
    # pairs we need markers to discriminate between, given this
    # parent node
    taxonomy_idx_array = _get_taxonomy_idx(
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node,
        marker_gene_array=marker_gene_array)

    # mask across all available taxonomy pairs indicating
    # which ones are actually being considered right now
    taxonomy_mask = np.zeros(
        marker_gene_array.n_pairs,
        dtype=bool)
    taxonomy_mask[taxonomy_idx_array] = True

    # figure out which genes are in both the reference dataset
    # and the query dataset
    matched_genes = match_genes(
        reference_gene_names=marker_gene_array.gene_names,
        query_gene_names=query_gene_names)

    reference_gene_mask = np.zeros(marker_gene_array.n_genes, dtype=bool)
    reference_gene_mask[matched_genes['reference']] = True
    if reference_gene_mask.sum() == 0:
        raise RuntimeError(
            "No gene overlap between reference and query set")

    # calculate the initial array indicating how useful each gene
    # (*all* reference genes at this point) is at discriminating
    # between the taxonomy pairs that we cair about
    utility_array = create_utility_array(
            cache_path=marker_gene_array.cache_path,
            gb_size=10,
            taxonomy_mask=taxonomy_idx_array)

    # mask out the genes which were not matched so that they
    # are not selected
    utility_array[np.logical_not(reference_gene_mask)] = 0

    # tally how many markers are chosen for each taxonomy pair
    # (the 2 columns are for up/down distinctions)
    marker_counts = np.zeros((len(taxonomy_idx_array), 2), dtype=np.uint8)
    been_filled = np.zeros((len(taxonomy_idx_array), 2), dtype=bool)
    been_filled_size = been_filled.size

    # the final result
    marker_gene_idx_set = set()

    # we will just start at the most useful gene and work our way down
    sorted_utility_idx = list(np.argsort(utility_array))

    duration = (time.time()-t0)/3600.0
    print(f"preparation took {duration:.2e} hours")

    while True:

        # because the utility_array for genes that are not in the query
        # set was set to zero at the beginning, this ought to indicate that
        # none of the genes left have any utility in the taxonomy pairs
        # we care about
        if utility_array.max() <= 0:
            break

        # chose the gene with the largest utility
        chosen_idx = sorted_utility_idx.pop(-1)
        if chosen_idx in marker_gene_idx_set:
            raise RuntimeError(
                f"Something is wrong; chose gene {chosen_idx} twice")
        marker_gene_idx_set.add(chosen_idx)

        # so we do not choose this gene again
        utility_array[chosen_idx] = 0

        # update marker_counts
        (marker_mask,
         up_mask) = marker_gene_array.marker_mask_from_gene_idx(
                         gene_idx=chosen_idx)

        marker_mask = marker_mask[taxonomy_mask]
        up_mask = up_mask[taxonomy_mask]

        full_mask = np.logical_and(marker_mask, up_mask)
        marker_counts[full_mask, 1] += 1
        full_mask = np.logical_and(marker_mask, np.logical_not(up_mask))
        marker_counts[full_mask, 0] += 1

        # see if we have completed the desired complement of genes
        # for any taxonomy pair
        newly_full = np.where(
            np.logical_and(
                np.logical_not(been_filled),
                marker_counts >= n_per_utility))

        # if so, update the utility_array so that taxonomy pairs that
        # already have their full complement of marker genes do not
        # contribute to the utility score if genes
        if len(newly_full[0]) > 0:
            for pair_idx, raw_sign in zip(newly_full[0], newly_full[1]):
                sign = {0: -1, 1: 1}[raw_sign]
                utility_array = recalculate_utility_array(
                    utility_array=utility_array,
                    marker_gene_array=marker_gene_array,
                    pair_idx=pair_idx,
                    sign=sign)
                been_filled[pair_idx, raw_sign] = True
            sorted_utility_idx = list(np.argsort(utility_array))
            filled_sum = been_filled.sum()
            duration = (time.time()-t0)/3600.0
            print(f"filled {filled_sum} of {been_filled_size} "
                  f"in {duration:.2e} hours -- "
                  f"{len(marker_gene_idx_set)} genes")

            if been_filled.sum() == been_filled_size:
                # we have found all the genes we need
                break

    marker_gene_names = [
        marker_gene_array.gene_names[idx]
        for idx in marker_gene_idx_set]
    marker_gene_names.sort()
    return marker_gene_names


def _get_taxonomy_idx(
        taxonomy_tree,
        parent_node,
        marker_gene_array):
    """
    Return the numpy array of indexes for the leaf nodes
    that need to be compared given the taxonomy tree
    and the specified parent node
    """
    leaf_pairs = get_all_leaf_pairs(
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node)

    taxonomy_idx_array = [
        marker_gene_array.idx_of_pair(
            leaf[0], leaf[1], leaf[2])
        for leaf in leaf_pairs]

    taxonomy_idx_array = np.array(
        taxonomy_idx_array)

    taxonomy_idx_array = np.sort(taxonomy_idx_array)
    return taxonomy_idx_array


def recalculate_utility_array(
        utility_array,
        marker_gene_array,
        pair_idx,
        sign):
    """
    utility_array is existing utility array

    marker_gene_array is a MarkerGeneArray

    pair_idx is the index of the taxonomy pair that has been fulfilled

    sign is (+1, -1), indicating which node in the pair is up-regulated
    in the gene


    Returns
    -------
    utility array with the necessary rows (those that were markers
    for the given (taxonomy_idx, sign) pair) decremented.

    Notes
    -----
    This method also alters utility_array in place
    """
    if sign not in (1, -1):
        raise RuntimeError(
            f"Unclear how to interpret sign = {sign}\n"
            "must be one of (-1, +1)")

    (marker_mask,
     up_mask) = marker_gene_array.marker_mask_from_pair_idx(
                 pair_idx=pair_idx)

    if sign > 0:
        full_mask = np.logical_and(marker_mask, up_mask)
    else:
        full_mask = np.logical_and(marker_mask, np.logical_not(up_mask))
    utility_array[full_mask] -= 1
    return utility_array
