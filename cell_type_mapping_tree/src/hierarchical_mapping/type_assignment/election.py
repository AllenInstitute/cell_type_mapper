import numpy as np
import time

from hierarchical_mapping.utils.distance_utils import (
    correlation_nearest_neighbors)


from hierarchical_mapping.type_assignment.matching import (
   get_leaf_means,
   assemble_query_data)


def run_type_assignment(
        full_query_gene_data,
        precomputed_stats_path,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor,
        bootstrap_iteration,
        rng):

    # get a dict mapping leaf node name
    # to mean gene expression profile
    leaf_node_lookup = get_leaf_means(
        taxonomy_tree=taxonomy_tree,
        precompute_path=precomputed_stats_path)

    # create effectively empty list of dicts to
    # store the hierarchical classification of
    # each cell in full_query_gene_data
    hierarchy = taxonomy_tree['hierarchy']
    result = []
    for i_cell in range(full_query_gene_data.shape[0]):
         this = dict()
         for level in hierarchy:
             this[level] = None
         result.append(this)

    # list of levels in the taxonomy (None means consider all clusters)
    level_list = [None] + list(hierarchy)

    # dict to keep track of which rows were assigned to which
    # child types in a rapidly-accessible way
    previously_assigned = dict()

    # loop over parent_level, assigning query cells to the child types
    # of that level
    t0 = time.time()
    for parent_level, child_level in zip(level_list[:-1], level_list[1:]):

        # build list of parent nodes to search from
        if parent_level is None:
            parent_node_list = [None]
        else:
            k_list = list(taxonomy_tree[parent_level].keys())
            k_list.sort()
            parent_node_list = []
            for k in k_list:
                parent_node_list.append((parent_level, k))

        previously_assigned[child_level] = dict()

        for parent_node in parent_node_list:

            # only consider the query cells that were assigned
            # to the current parent
            if parent_node is None:
                chosen_idx = np.arange(full_query_gene_data.shape[0])
                chosen_query_data = full_query_gene_data
            else:
                if parent_node[1] in previously_assigned[parent_level]:
                    chosen_idx = previously_assigned[parent_level][parent_node[1]]
                    chosen_query_data = full_query_gene_data[chosen_idx, :]
                else:
                    chosen_idx = []

            if len(chosen_idx) == 0:
                continue

            assignment = _run_type_assignment(
                            full_query_gene_data=chosen_query_data,
                            leaf_node_lookup=leaf_node_lookup,
                            marker_gene_cache_path=marker_gene_cache_path,
                            taxonomy_tree=taxonomy_tree,
                            parent_node=parent_node,
                            bootstrap_factor=bootstrap_factor,
                            bootstrap_iteration=bootstrap_iteration,
                            rng=rng)

            # popuulate the dict keeping track of the rows in
            # full_query_gene_data that were assigned to each
            # possible child type
            type_to_idx = dict()
            idx_to_type = []
            for idx, celltype in enumerate(set(assignment)):
                type_to_idx[celltype] = idx
                idx_to_type.append(celltype)
            assignment_idx = np.array([type_to_idx[celltype]
                                       for celltype in assignment])

            for idx in range(len(idx_to_type)):
                celltype = idx_to_type[idx]
                assigned_this = (assignment_idx==idx)
                assigned_this= chosen_idx[assigned_this]
                previously_assigned[child_level][celltype] = assigned_this

            # assign cells to their chosen child_level nodes
            for i_cell, assigned_type in zip(chosen_idx, assignment):
                result[i_cell][child_level] = assigned_type
        duration = (time.time()-t0)/60.0
        print(f"assigned {parent_level} after {duration:.2e} minutes")

    return result


def _run_type_assignment(
        full_query_gene_data,
        leaf_node_lookup,
        marker_gene_cache_path,
        taxonomy_tree,
        parent_node,
        bootstrap_factor,
        bootstrap_iteration,
        rng):

    query_data = assemble_query_data(
        full_query_data=full_query_gene_data,
        mean_profile_lookup=leaf_node_lookup,
        marker_cache_path=marker_gene_cache_path,
        taxonomy_tree=taxonomy_tree,
        parent_node=parent_node)

    result = choose_node(
        query_gene_data=query_data['query_data'],
        reference_gene_data=query_data['reference_data'],
        reference_types=query_data['reference_types'],
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    return result


def choose_node(
         query_gene_data,
         reference_gene_data,
         reference_types,
         bootstrap_factor,
         bootstrap_iteration,
         rng):
    """
    Parameters
    ----------
    query_gene_data
        cell-by-marker-gene array of query data
    reference_gene_data
        cell-by-marker-gene array of reference data
    reference_types
        array of cell types we are chosing from (n_cells in size)
    bootstrap_factor
        Factor by which to subsample reference genes at each bootstrap
    bootstrap_iteration
        Number of bootstrapping iterations
    rng
        random number generator

    Returns
    -------
    Array of cell type assignments (majority rule)
    """

    votes = tally_votes(
        query_gene_data=query_gene_data,
        reference_gene_data=reference_gene_data,
        bootstrap_factor=bootstrap_factor,
        bootstrap_iteration=bootstrap_iteration,
        rng=rng)

    chosen_type = np.argmax(votes, axis=1)
    result = [reference_types[ii] for ii in chosen_type]
    return np.array(result)


def tally_votes(
         query_gene_data,
         reference_gene_data,
         bootstrap_factor,
         bootstrap_iteration,
         rng):
    """
    Parameters
    ----------
    query_gene_data
        cell-by-marker-gene array of query data
    reference_gene_data
        cell-by-marker-gene array of reference data
    reference_types
        array of cell types we are chosing from (n_cells in size)
    bootstrap_factor
        Factor by which to subsample reference genes at each bootstrap
    bootstrap_iteration
        Number of bootstrapping iterations
    rng
        random number generator

    Returns
    -------
    Array of ints. Each row is a query cell. Each column is a
    reference cell. The value is how many iterations voted for
    "this query cell is the same type as this reference cell"
    """
    n_markers = query_gene_data.shape[1]
    marker_idx = np.arange(n_markers)
    n_bootstrap = np.round(bootstrap_factor*n_markers).astype(int)

    votes = np.zeros((query_gene_data.shape[0], reference_gene_data.shape[0]),
                     dtype=int)

    # query_idx is needed to associate each vote with its row
    # in the votes array
    query_idx = np.arange(query_gene_data.shape[0])

    for i_iteration in range(bootstrap_iteration):
        chosen_idx = rng.choice(marker_idx, n_bootstrap, replace=False)
        chosen_idx = np.sort(chosen_idx)
        bootstrap_query = query_gene_data[:, chosen_idx]
        bootstrap_reference = reference_gene_data[:, chosen_idx]

        nearest_neighbors = correlation_nearest_neighbors(
            baseline_array=bootstrap_reference,
            query_array=bootstrap_query)

        votes[query_idx, nearest_neighbors] += 1
    return votes
