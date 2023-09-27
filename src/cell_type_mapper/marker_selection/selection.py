import numpy as np

from cell_type_mapper.utils.multiprocessing_utils import (
    DummyLock)

from cell_type_mapper.marker_selection.utils import (
    create_utility_array)

from cell_type_mapper.marker_selection.marker_array_utils import (
    thin_marker_gene_array_by_gene)


def select_marker_genes_v2(
        marker_gene_array,
        query_gene_names,
        taxonomy_tree,
        parent_node,
        n_per_utility=15,
        lock=None,
        summary_log=None,
        tmp_dir=None):
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
        instance of
        cell_type_mapper.taxonomty.taxonomy_tree.TaxonomyTree
        encoding the taxonomy tree

    parent_node:
        (level, node) tuple indicating the parent for whose
        children we are selecting markers (None will indicate
        that we are at the root of the tree)

    n_per_utility:
        Number of marker genes to select per (node1, node2, +/-) set
        (+/- indicates which node the gene is more prominent in)

    lock:
       Optional multiprocessing lock to prevent stdout prints from
       stumbling over each other (can be None)

    summary_log:
        If not None, a dict-like object (probably a
        multiprocessing.Manager.dict) mapping parent node
        to a summary of the performance of marker
        selection on that node).

    tmp_dir:
        Directory for storing scratch files so big matrix manipulations
        do not happen in memory.

    Returns
    -------
    A list of marker gene names.
        (Alphabetized for lack of a better ordering scheme.)

    A string summarizing how well the marker selection did.
    """

    if lock is None:
        lock = DummyLock()

    marker_gene_array = thin_marker_gene_array_by_gene(
        marker_gene_array=marker_gene_array,
        query_gene_names=query_gene_names,
        tmp_dir=tmp_dir)

    # get a numpy array of indices indicating which taxonomy
    # pairs we need markers to discriminate between, given this
    # parent node
    taxonomy_idx_array = _get_taxonomy_idx(
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node,
        marker_gene_array=marker_gene_array)

    # calculate the initial array indicating how useful each gene
    # (*all* reference genes at this point) is at discriminating
    # between the taxonomy pairs that we cair about
    (utility_array,
     marker_census) = create_utility_array(
            marker_gene_array=marker_gene_array,
            gb_size=10,
            taxonomy_mask=taxonomy_idx_array)

    (marker_gene_names,
     summary_log_message) = _run_selection(
        marker_gene_array=marker_gene_array,
        utility_array=utility_array,
        marker_census=marker_census,
        taxonomy_idx_array=taxonomy_idx_array,
        n_per_utility=n_per_utility,
        parent_node=parent_node,
        lock=lock)

    if summary_log is not None:
        if parent_node is None:
            log_key = 'None'
        else:
            log_key = f'{parent_node[0]}/{parent_node[1]}'
        summary_log[log_key] = summary_log_message

    return marker_gene_names


def _run_selection(
        marker_gene_array,
        utility_array,
        marker_census,
        taxonomy_idx_array,
        n_per_utility,
        parent_node,
        lock=None):

    # how many total marker genes were there originally
    # (for logging purposes)
    n_useful_0 = (utility_array > 0).sum()

    if lock is None:
        lock = DummyLock()

    # the final result
    marker_gene_idx_set = set()
    marker_gene_name_list = []  # in order they were chosen

    # tally how many markers are chosen for each taxonomy pair
    # (the 2 columns are for up/down distinctions)
    marker_counts = np.zeros((len(taxonomy_idx_array), 2), dtype=np.uint8)
    been_filled = np.zeros((len(taxonomy_idx_array), 2), dtype=bool)
    been_filled_size = been_filled.size

    # first pass (will mostly mark off those taxon pair that
    # have zero markers)
    (been_filled,
     utility_array,
     sorted_utility_idx) = _update_been_filled(
             marker_counts=marker_counts,
             been_filled=been_filled,
             utility_array=utility_array,
             marker_census=marker_census,
             sorted_utility_idx=None,
             n_per_utility=n_per_utility,
             marker_gene_array=marker_gene_array,
             taxonomy_idx_array=taxonomy_idx_array)

    filled_sum = been_filled.sum()

    (marker_gene_idx_set,
     marker_gene_name_list,
     utility_array,
     sorted_utility_idx,
     marker_counts) = _choose_desperate_markers(
        marker_gene_idx_set=marker_gene_idx_set,
        marker_gene_name_list=marker_gene_name_list,
        utility_array=utility_array,
        sorted_utility_idx=sorted_utility_idx,
        marker_gene_array=marker_gene_array,
        marker_counts=marker_counts,
        taxonomy_idx_array=taxonomy_idx_array,
        marker_census=marker_census,
        n_per_utility=n_per_utility,
        n_desperate=n_per_utility)

    n_desperate = len(marker_gene_name_list)

    broke_because = ''
    while True:

        (been_filled,
         utility_array,
         sorted_utility_idx) = _update_been_filled(
                 marker_counts=marker_counts,
                 been_filled=been_filled,
                 utility_array=utility_array,
                 marker_census=marker_census,
                 sorted_utility_idx=sorted_utility_idx,
                 n_per_utility=n_per_utility,
                 marker_gene_array=marker_gene_array,
                 taxonomy_idx_array=taxonomy_idx_array)

        # because the utility_array for genes that are not in the query
        # set was set to zero at the beginning, this ought to indicate that
        # none of the genes left have any utility in the taxonomy pairs
        # we care about
        if utility_array.max() <= 0:
            broke_because = 'utility_array.max()'
            break

        filled_sum = been_filled.sum()
        if filled_sum == been_filled_size:
            # we have found all the genes we need
            broke_because = 'been_filled.sum()'
            break

        (marker_gene_idx_set,
         marker_gene_name_lit,
         utility_array,
         sorted_utility_idx,
         marker_counts) = _choose_gene(
             marker_gene_idx_set=marker_gene_idx_set,
             marker_gene_name_list=marker_gene_name_list,
             utility_array=utility_array,
             sorted_utility_idx=sorted_utility_idx,
             marker_gene_array=marker_gene_array,
             marker_counts=marker_counts,
             taxonomy_idx_array=taxonomy_idx_array,
             chosen_idx=None)

    assert len(marker_gene_idx_set) == len(marker_gene_name_list)

    stat_msg, stat_dict = _stats_from_marker_counts(marker_counts)
    stat_dict['n_genes'] = len(marker_gene_name_list)
    stat_dict['filled'] = int(been_filled.sum())
    stat_dict['unfilled'] = int(been_filled_size)-stat_dict['filled']
    stat_dict['n_desperate'] = int(n_desperate)
    stat_dict['n_original_markers'] = int(n_useful_0)

    # how many taxon pairs have fewer than n_th markers in
    # the 'up' and 'down' regulated slots
    marker_dist = dict()

    n_th_values = list(range(5, n_per_utility, 5))
    n_th_values = [1] + n_th_values
    if (n_per_utility) not in n_th_values:
        n_th_values.append(n_per_utility)
    n_th_values.sort()

    for n_th in n_th_values:
        fewer_down = (marker_counts[:, 0] < n_th).sum()
        fewer_up = (marker_counts[:, 1] < n_th).sum()
        marker_dist[f'lt_{n_th}'] = {
            'up': int(fewer_up),
            'down': int(fewer_down)}

    stat_dict['marker_distribution'] = marker_dist

    msg = f"\n======parent_node: {parent_node}======\n"
    msg += f"selected {len(marker_gene_name_list)} from "
    msg += f"{marker_gene_array.n_genes}\n"
    msg += f"filled {been_filled.sum()} of {been_filled_size}\n"
    msg += f"broke because {broke_because}\n"
    msg += stat_msg
    msg += "\n============"
    with lock:
        print(msg)

    return marker_gene_name_list, stat_dict


def _choose_gene(
        marker_gene_idx_set,
        marker_gene_name_list,
        utility_array,
        sorted_utility_idx,
        marker_gene_array,
        marker_counts,
        taxonomy_idx_array,
        chosen_idx=None):

    # chose the gene with the largest utility
    if chosen_idx is None:
        chosen_idx = sorted_utility_idx.pop(-1)
    else:
        for ii in range(len(sorted_utility_idx)):
            if sorted_utility_idx[ii] == chosen_idx:
                to_pop = ii
                break
        sorted_utility_idx.pop(to_pop)

    marker_gene_name_list.append(marker_gene_array.gene_names[chosen_idx])

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
        marker_counts=marker_counts)

    return (marker_gene_idx_set,
            marker_gene_name_list,
            utility_array,
            sorted_utility_idx,
            marker_counts)


def _choose_desperate_markers(
        marker_gene_idx_set,
        marker_gene_name_list,
        utility_array,
        sorted_utility_idx,
        marker_gene_array,
        marker_counts,
        taxonomy_idx_array,
        marker_census,
        n_per_utility,
        n_desperate=5):
    """
    Find cases where a taxonomy_pair cannot match 2*n_per_utility
    markers. Select all of the markers for those genes.
    """
    total_markers = marker_census.sum(axis=1)
    desperate_cases = np.where(np.logical_and(
            total_markers <= n_desperate,
            total_markers > 0))[0]

    for local_idx in desperate_cases:
        global_idx = taxonomy_idx_array[local_idx]
        (marker_mask,
         _) = marker_gene_array.marker_mask_from_pair_idx(
                 pair_idx=global_idx)
        valid_genes = np.where(marker_mask)[0]
        for gene_idx in valid_genes:
            if gene_idx in marker_gene_idx_set:
                continue

            (marker_gene_idx_set,
             marker_gene_name_list,
             utility_array,
             sorted_utility_idx,
             marker_counts) = _choose_gene(
                    marker_gene_idx_set=marker_gene_idx_set,
                    marker_gene_name_list=marker_gene_name_list,
                    utility_array=utility_array,
                    sorted_utility_idx=sorted_utility_idx,
                    marker_gene_array=marker_gene_array,
                    marker_counts=marker_counts,
                    taxonomy_idx_array=taxonomy_idx_array,
                    chosen_idx=gene_idx)

    return (marker_gene_idx_set,
            marker_gene_name_list,
            utility_array,
            sorted_utility_idx,
            marker_counts)


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

    # which taxons can even hope to fill n_per_utility
    # markers in both directions
    are_possible = (marker_census >= n_per_utility)
    are_possible = are_possible.sum(axis=1)
    are_possible = (are_possible == 2)

    # see if we have completed the desired complement of genes
    # for any taxonomy pair
    raw_full_mask = (marker_counts >= n_per_utility)
    newly_full_mask = np.copy(raw_full_mask)

    # only flag those for which it was possible to fill it
    newly_full_mask[:, 0] = np.logical_and(newly_full_mask[:, 0], are_possible)
    newly_full_mask[:, 1] = np.logical_and(newly_full_mask[:, 1], are_possible)

    # check cases where we have grabbed all the markers we can
    maxed_out = (marker_counts == marker_census)

    # also cases where we have the total number of desired markers
    # for the taxon pair, regardless of their up/down distribution
    tot_counts = marker_counts.sum(axis=1)
    tot_maxed = (tot_counts >= 2*n_per_utility)
    maxed_out[:, 0] = np.logical_or(maxed_out[:, 0], tot_maxed)
    maxed_out[:, 1] = np.logical_or(maxed_out[:, 1], tot_maxed)

    newly_full_mask = np.logical_or(
        newly_full_mask,
        maxed_out)

    # don't correct for pairs that were already marked
    # as "filled"
    newly_full_mask = np.logical_and(
        newly_full_mask,
        np.logical_not(been_filled))

    newly_full = np.where(newly_full_mask)

    # if so, update the utility_array so that taxonomy pairs that
    # already have their full complement of marker genes do not
    # contribute to the utility score of genes
    if len(newly_full[0]) > 0:
        for pair_idx, raw_sign in zip(newly_full[0], newly_full[1]):
            sign = {0: -1, 1: 1}[raw_sign]
            utility_array = recalculate_utility_array(
                utility_array=utility_array,
                marker_gene_array=marker_gene_array,
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
    leaf_pairs = taxonomy_tree.leaves_to_compare(
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

    if sign > 0:
        full_mask = marker_gene_array.up_mask_from_pair_idx(
                            pair_idx=pair_idx)
    else:
        full_mask = marker_gene_array.down_mask_from_pair_idx(
                            pair_idx=pair_idx)
    utility_array[full_mask] -= 1
    return utility_array


def _stats_from_marker_counts(
        marker_counts):

    genes_per_pair = marker_counts.sum(axis=1)

    # these stats are by pair, *not* by utility set
    # (i.e. up_ and down_regulated markers are lumped together
    # for the given taxon pairs)
    med_genes = np.median(genes_per_pair)
    min_genes = genes_per_pair.min()
    max_genes = genes_per_pair.max()
    n_zero = (genes_per_pair == 0).sum()
    lt_5 = (genes_per_pair < 5).sum()
    lt_15 = (genes_per_pair < 15).sum()
    lt_30 = (genes_per_pair < 30).sum()
    msg = f"genes per pair {min_genes} {med_genes} {max_genes} "
    msg += f"n_zero {n_zero:.2e} n_lt_5 {lt_5:.2e} "
    msg += f"n_lt15 {lt_15:.2e} n_lt30 {lt_30:.2e}"

    as_dict = {
        'min_n_genes': int(min_genes),
        'median_n_genes': int(med_genes),
        'max_n_genes': int(max_genes),
        'n_zero': int(n_zero),
        'lt_5': int(lt_5),
        'lt_15': int(lt_15),
        'lt_30': int(lt_30)}

    return msg, as_dict
