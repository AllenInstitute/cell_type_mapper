import multiprocessing

from hierarchical_mapping.utils.taxonomy_utils import (
    get_all_leaf_pairs)

from hierarchical_mapping.utils.multiprocessing_utils import (
    winnow_process_dict)

from hierarchical_mapping.marker_selection.selection import (
    select_marker_genes_v2)

from hierarchical_mapping.marker_selection.marker_array import (
    MarkerGeneArray)


def select_all_markers(
        marker_cache_path,
        query_gene_names,
        taxonomy_tree,
        n_per_utility=15,
        n_processors=4,
        behemoth_cutoff=1000000):
    """
    Select all of the markers necessary for a taxonomy.
    Save them as a JSONized dict (for now).

    Parameters
    ----------
    marker_cache_path:
        Path to the HDF5 file with the raw maker data from the
        reference dataset.
    query_gene_names:
        List of gene names from the query dataset
    taxonomy_tree:
        Dict representing the taxonomy tree.
    n_per_utility:
        How many genes to select per (taxon_pair, sign)
        combination
    n_processors:
        Number of independent workers to spin up.
    behemoth_cutoff:
        Number of leaf nodes for a parent to be considered
        a behemoth

    Returns
    -------
    A dict mapping parent node tuple to list of marker gene
    names
    """

    parent_list = [None]
    n_leaves = [len(get_all_leaf_pairs(taxonomy_tree=taxonomy_tree,
                                       parent_node=None))]
    for level in taxonomy_tree['hierarchy'][:-1]:
        for node in taxonomy_tree[level]:
            parent = (level, node)
            parent_list.append(parent)
            n_leaves.append(
                len(get_all_leaf_pairs(
                        taxonomy_tree=taxonomy_tree,
                        parent_node=parent)))

    # want to make sure that the memory hogs are not
    # all running at once
    behemoth_parents = []
    smaller_parents = []
    for parent, n_leaf in zip(parent_list, n_leaves):
        if n_leaf > behemoth_cutoff:
            behemoth_parents.append(parent)
        else:
            smaller_parents.append(parent)

    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    input_lock = mgr.Lock()
    stdout_lock = mgr.Lock()

    started_parents = set()
    completed_parents = set()
    process_dict = dict()
    while len(started_parents) < len(parent_list):

        are_behemoths_running = False
        for p in behemoth_parents:
            if p in started_parents and p not in completed_parents:
                are_behemoths_running = True
                break

        have_chosen_parent = False
        if not are_behemoths_running:
            for parent in behemoth_parents:
                if parent not in started_parents:
                    chosen_parent = parent
                    have_chosen_parent = True
                    break
        if not have_chosen_parent:
            for parent in smaller_parents:
                if parent not in started_parents:
                    chosen_parent = parent
                    have_chosen_parent = True
                    break

        if have_chosen_parent:
            started_parents.add(chosen_parent)
            p = multiprocessing.Process(
                    target=_marker_selection_worker,
                    kwargs={
                        'marker_cache_path': marker_cache_path,
                        'query_gene_names': query_gene_names,
                        'taxonomy_tree': taxonomy_tree,
                        'parent_node': chosen_parent,
                        'n_per_utility': n_per_utility,
                        'behemoth_cutoff': behemoth_cutoff,
                        'output_dict': output_dict,
                        'input_lock': input_lock,
                        'stdout_lock': stdout_lock})
            p.start()

            process_dict[chosen_parent] = p

        # the test on have_chosen_parent is there in case we have
        # a traffic jam of behemoths trying to get through
        while len(process_dict) >= n_processors or not have_chosen_parent:
            k0 = set(process_dict.keys())
            process_dict = winnow_process_dict(process_dict)
            k1 = set(process_dict.keys())
            if len(k1) < len(k0):
                completed_parents = completed_parents.union(k0-k1)
                have_chosen_parent = True

    while len(process_dict) > 0:
        process_dict = winnow_process_dict(process_dict)

    return dict(output_dict)


def _marker_selection_worker(
        marker_cache_path,
        query_gene_names,
        taxonomy_tree,
        parent_node,
        behemoth_cutoff,
        n_per_utility,
        output_dict,
        input_lock,
        stdout_lock):

    leaf_pair_list = get_all_leaf_pairs(
            taxonomy_tree=taxonomy_tree,
            parent_node=parent_node)

    # this could happen if a parent node has only one
    # immediate descendant
    if len(leaf_pair_list) == 0:
        output_dict[parent_node] = []
        return

    if len(leaf_pair_list) < behemoth_cutoff:
        only_keep_pairs = leaf_pair_list
    else:
        only_keep_pairs = None

    with input_lock:
        marker_gene_array = MarkerGeneArray(
            cache_path=marker_cache_path,
            only_keep_pairs=only_keep_pairs)

    marker_genes = select_marker_genes_v2(
        marker_gene_array=marker_gene_array,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node,
        n_per_utility=n_per_utility,
        lock=stdout_lock)

    output_dict[parent_node] = marker_genes
