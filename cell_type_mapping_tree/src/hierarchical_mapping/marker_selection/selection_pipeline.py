import multiprocessing

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
        instance of
        hierarchical_mapping.taxonomty.taxonomy_tree.TaxonomyTree
        encoding the taxonomy tree
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

    parent_to_leaves = dict()
    parent_list = taxonomy_tree.all_parents

    # want to make sure that the memory hogs are not
    # all running at once
    behemoth_parents = []
    smaller_parents = []
    for parent in parent_list:
        leaves = taxonomy_tree.leaves_to_compare(
            parent_node=parent)
        parent_to_leaves[parent] = leaves
        n_leaves = len(leaves)
        if n_leaves > behemoth_cutoff:
            behemoth_parents.append(parent)
        else:
            smaller_parents.append(parent)

    parent_marker_cache = MarkerGeneArray.from_cache_path(
        cache_path=marker_cache_path)

    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
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
        is_behemoth = False
        if not are_behemoths_running:
            for parent in behemoth_parents:
                if parent not in started_parents:
                    chosen_parent = parent
                    have_chosen_parent = True
                    is_behemoth = True
                    break
        if not have_chosen_parent:
            for parent in smaller_parents:
                if parent not in started_parents:
                    chosen_parent = parent
                    have_chosen_parent = True
                    break

        if have_chosen_parent:
            started_parents.add(chosen_parent)
            leaves = parent_to_leaves[chosen_parent]
            if len(leaves) == 0:
                output_dict[chosen_parent] = []
                completed_parents.add(chosen_parent)
            else:
                if is_behemoth:
                    marker_gene_array = parent_marker_cache
                else:
                    marker_gene_array = \
                        parent_marker_cache.downsample_pairs_to_other(
                            only_keep_pairs=leaves)

                p = multiprocessing.Process(
                        target=_marker_selection_worker,
                        kwargs={
                            'marker_gene_array': marker_gene_array,
                            'query_gene_names': query_gene_names,
                            'taxonomy_tree': taxonomy_tree,
                            'parent_node': chosen_parent,
                            'n_per_utility': n_per_utility,
                            'output_dict': output_dict,
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
        marker_gene_array,
        query_gene_names,
        taxonomy_tree,
        parent_node,
        n_per_utility,
        output_dict,
        stdout_lock):

    leaf_pair_list = taxonomy_tree.leaves_to_compare(
        parent_node=parent_node)

    # this could happen if a parent node has only one
    # immediate descendant
    if len(leaf_pair_list) == 0:
        output_dict[parent_node] = []
        return

    marker_genes = select_marker_genes_v2(
        marker_gene_array=marker_gene_array,
        query_gene_names=query_gene_names,
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node,
        n_per_utility=n_per_utility,
        lock=stdout_lock)

    output_dict[parent_node] = marker_genes
