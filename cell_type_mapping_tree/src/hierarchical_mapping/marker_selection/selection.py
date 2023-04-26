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

    marker_gene_array = _thin_marker_gene_array(
        marker_gene_array=marker_gene_array,
        query_gene_names=query_gene_names)

    # calculate the initial array indicating how useful each gene
    # (*all* reference genes at this point) is at discriminating
    # between the taxonomy pairs that we cair about
    (utility_array,
     marker_census,
     taxon_scores) = create_utility_array(
            marker_gene_array=marker_gene_array,
            gb_size=30,
            taxonomy_mask=taxonomy_idx_array)

    census_sum = np.sum(marker_census, axis=1)
    min_dex = np.argmin(census_sum)
    print(f"min census {census_sum.min()}")
    print(f"at {min_dex}")
    print(f"which is taxon pair {taxonomy_idx_array[min_dex]}")

    duration = (time.time()-t0)/3600.0
    print(f"preparation took {duration:.2e} hours")

    marker_gene_names = _run_selection(
        marker_gene_array=marker_gene_array,
        utility_array=utility_array,
        marker_census=marker_census,
        taxon_scores=taxon_scores,
        taxonomy_idx_array=taxonomy_idx_array,
        n_per_utility=n_per_utility)

    return marker_gene_names


def _run_selection(
        marker_gene_array,
        utility_array,
        marker_census,
        taxon_scores,
        taxonomy_idx_array,
        n_per_utility):

    t0 = time.time()

    # the final result
    marker_gene_idx_set = set()

    # tally how many markers are chosen for each taxonomy pair
    # (the 2 columns are for up/down distinctions)
    marker_counts = np.zeros((len(taxonomy_idx_array), 2), dtype=np.uint8)
    been_filled = np.zeros((len(taxonomy_idx_array), 2), dtype=bool)
    been_filled_size = been_filled.size

    sorted_utility_idx = None
    filled_sum = been_filled.sum()

    while True:

        # because the utility_array for genes that are not in the query
        # set was set to zero at the beginning, this ought to indicate that
        # none of the genes left have any utility in the taxonomy pairs
        # we care about
        if utility_array.max() <= 0:
            break

        filled_sum0 = filled_sum

        (been_filled,
         utility_array,
         sorted_utility_idx) = _update_been_filled(
                 marker_counts=marker_counts,
                 been_filled=been_filled,
                 utility_array=utility_array,
                 marker_census=marker_census,
                 taxon_scores=taxon_scores,
                 sorted_utility_idx=sorted_utility_idx,
                 n_per_utility=n_per_utility,
                 marker_gene_array=marker_gene_array,
                 taxonomy_idx_array=taxonomy_idx_array)

        filled_sum = been_filled.sum()
        if filled_sum > filled_sum0:
            duration = (time.time()-t0)/3600.0
            print(f"filled {filled_sum} of {been_filled_size} "
                  f"in {duration:.2e} hours -- "
                  f"{len(marker_gene_idx_set)} genes -- "
                  + _stats_from_marker_counts(marker_counts)
                  + f" max_utility {utility_array.max():.2e}")

        if been_filled.sum() == been_filled_size:
            # we have found all the genes we need
            break

        # chose the gene with the largest utility
        chosen_idx = sorted_utility_idx.pop(-1)
        if chosen_idx in marker_gene_idx_set:
            raise RuntimeError(
                f"Something is wrong; chose gene {chosen_idx} twice")
        marker_gene_idx_set.add(chosen_idx)

        # so we do not choose this gene again
        utility_array[chosen_idx] = -1.0

        # update marker_counts
        marker_counts = _update_marker_counts(
            marker_gene_array=marker_gene_array,
            chosen_gene_idx=chosen_idx,
            taxonomy_idx_array=taxonomy_idx_array,
            marker_counts=marker_counts,)

    marker_gene_names = [
        marker_gene_array.gene_names[idx]
        for idx in marker_gene_idx_set]
    marker_gene_names.sort()
    print(f"selected {len(marker_gene_names)} from "
          f"{marker_gene_array.n_genes}")
    print(f"filled {been_filled.sum()} of {been_filled_size}")
    print(_stats_from_marker_counts(marker_counts))
    return marker_gene_names


def _thin_marker_gene_array(
        marker_gene_array,
        query_gene_names):
    """
    Remove rows that are not in the query gene set from
    marker_gene_array

    Parameters
    ----------
    marker_gene_array:
        A MarkerGeneArray containing the marker gene data
        from the reference dataset
    query_gene_names:
        List of the names of the genes in the query dataset

    Returns
    -------
    marker_gene_array:
        With only the nodes that overlap with query_gene_naems
        returned.
    """
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

    reference_gene_idx = np.where(reference_gene_mask)[0]
    marker_gene_array.downsample_genes(reference_gene_idx)
    return marker_gene_array


def _update_marker_counts(
        marker_gene_array,
        chosen_gene_idx,
        taxonomy_idx_array,
        marker_counts):
    """
    Update marker_counts to reflect the new chosen gene

    Parameters
    ----------
    marker_gene_array:
        A MarkerGeneArray carrying the marker data from the reference
        dataset
    chosen_gene_idx:
        The index of the gene that has been chosen as a the next marker
    taxonomy_idx_array:
        Array of pair_idx indicating which taxonomy pairs we need
        to contrast.
    marker_counts:
        A (n_pairs, 2) array indicating how many genes have been
        chosen for each (taxonomy_pair, sign) combination

    Returns
    -------
    marker_counts
        updated to reflect the newly chosen gene
    """
    # update marker_counts
    (marker_mask,
     up_mask) = marker_gene_array.marker_mask_from_gene_idx(
                     gene_idx=chosen_gene_idx)

    marker_mask = marker_mask[taxonomy_idx_array]
    up_mask = up_mask[taxonomy_idx_array]

    full_mask = np.logical_and(marker_mask, up_mask)
    marker_counts[full_mask, 1] += 1
    full_mask = np.logical_and(marker_mask, np.logical_not(up_mask))
    marker_counts[full_mask, 0] += 1

    return marker_counts


def _update_been_filled(
        marker_counts,
        been_filled,
        utility_array,
        marker_census,
        taxon_scores,
        sorted_utility_idx,
        n_per_utility,
        marker_gene_array,
        taxonomy_idx_array,
        n_min=5):
    """
    Update stats on which (taxonoy pair, sign) combinations
    have been filled.

    Parameters
    ----------
    marker_counts:
        (n_pairs, 2) array indicating how many genes have been
        selected for each (taxonomy_pair, sign) combination
    been_filled:
        (n_pairs, 2) array of booleans indicating which
        (taxonomy_pair, sign) combinations have had their
        complement of markers filled
    utility_array:
        (n_genes, ) array of integers indicating how many
        (taxonomy_pair, sign) combinations each gene is a
        marker for
    marker_census:
        (n_pairs, 2) array of integers indicating how many
        markers can be expected for each (taxon_pair, sign)
        combination
    taxon_scores
        (n_pairs, 2) array of utility scores given to each
        (taxonomy_pair, sign) combination
    sorted_utility_idx:
        Sorted indices of utility_array
    n_per_utility:
        number of genes to select per (taxonomy_pair, sign)
        combination
    marker_gene_array:
        A MarkerGeneArray carrying marker data form the
        reference dataset
    taxonomy_idx_array:
        The array of integers indicating which taxon pairs
        we are actually working with
    n_min:
        keep trying until we get at least this many markers
        per pair

    Returns
    -------
    been_filled:
        updated for any new combinations that have their complement
        filled
    utility_array:
        updated
    sorted_utility_array_idx:
        sorted idices of the utility array
    """

    # see if we have completed the desired complement of genes
    # for any taxonomy pair
    newly_full_mask = (marker_counts >= n_per_utility)

    # check cases where we have grabbed all the markers we can
    maxed_out = (marker_counts == marker_census)

    newly_full_mask = np.logical_or(
        newly_full_mask,
        maxed_out)

    # check hopeless cases
    is_hopeless = (marker_census < n_per_utility)
    filled_hopeless = np.logical_and(
        is_hopeless,
        marker_counts >= n_min)

    newly_full_mask = np.logical_or(
        newly_full_mask,
        filled_hopeless)

    # grab taxons where there are 2*n_per_utility markers
    # for the whole pair (+ and -) and at least a quarter
    # are in both (+ and -)
    de_facto_pair = (marker_counts.sum(axis=1) >= (2*n_per_utility))
    halfway_there = (marker_counts >= (n_per_utility//2))
    halfway_there = np.logical_and(halfway_there[:, 0],
                                   halfway_there[:, 1])
    de_facto_pair = np.logical_and(
        halfway_there,
        de_facto_pair)

    n0 = newly_full_mask.sum()
    newly_full_mask = np.logical_or(
        newly_full_mask,
        np.array([de_facto_pair, de_facto_pair]).transpose())
    print(f"de facto added {newly_full_mask.sum()-n0}")

    # don't correct for pairs that were already marked
    # as "filled"
    newly_full_mask = np.logical_and(
        newly_full_mask,
        np.logical_not(been_filled))

    newly_full = np.where(newly_full_mask)

    # if so, update the utility_array so that taxonomy pairs that
    # already have their full complement of marker genes do not
    # contribute to the utility score if genes
    if len(newly_full[0]) > 0:
        for pair_idx, raw_sign in zip(newly_full[0], newly_full[1]):
            sign = {0: -1, 1: 1}[raw_sign]
            this_score = taxon_scores[pair_idx, raw_sign]
            utility_array = recalculate_utility_array(
                utility_array=utility_array,
                marker_gene_array=marker_gene_array,
                taxon_score=this_score,
                pair_idx=taxonomy_idx_array[pair_idx],
                sign=sign)
            been_filled[pair_idx, raw_sign] = True

    if len(newly_full[0]) > 0 or sorted_utility_idx is None:
        sorted_utility_idx = list(np.argsort(utility_array))

    return (been_filled, utility_array, sorted_utility_idx)


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
        taxon_score,
        pair_idx,
        sign):
    """
    utility_array is existing utility array

    marker_gene_array is a MarkerGeneArray

    taxon_score is the utility score of the (taxon pair, sign)
    combination

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
    utility_array[full_mask] -= taxon_score
    return utility_array


def _stats_from_marker_counts(
        marker_counts):
    genes_per_pair = marker_counts.sum(axis=1)
    med_genes = np.median(genes_per_pair)
    min_genes = genes_per_pair.min()
    max_genes = genes_per_pair.max()
    n_zero = (genes_per_pair == 0).sum()
    lt_5 = (genes_per_pair < 5).sum()
    lt_15 = (genes_per_pair < 10).sum()
    lt_30 = (genes_per_pair < 30).sum()
    msg = f"genes per pair {min_genes} {med_genes} {max_genes} "
    msg += f"n_zero {n_zero:.2e} n_lt_5 {lt_5:.2e} "
    msg += f"n_lt15 {lt_15:.2e} n_lt30 {lt_30:.2e}"
    return msg
