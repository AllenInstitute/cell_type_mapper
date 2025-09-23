import numpy as np

import cell_type_mapper.utils.utils as mapper_utils
import cell_type_mapper.utils.distance_utils as distance_utils
import cell_type_mapper.type_assignment.matching as matching




def hann_tally_votes(
        full_query_gene_data,
        leaf_node_matrix,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor_lookup,
        bootstrap_iteration,
        rng,):
    """
    Marker lookup can be in terms of gene identifiers
    """

    data = matching.assemble_query_data_hann(
        full_query_data=full_query_data,
        mean_profile_matrix=leaf_node_matrix,
        taxonomy_tree=taxonomy_tree,
        marker_cache_path=marker_gene_cache_path
    )

    result_shape = (
        data['query_data'].n_cells,
        data['reference_data'].n_cells
    )

    vote_type = mapper_utils.choose_int_dtype((0, bootstrap_iteration))

    votes = np.zeros(
        result_shape,
        dtype=vote_type
    )
    corr = np.zeros(
        result_shape,
        dtype=float
    )

    child_to_parent = {
        leaf: taxonomy_tree.parents(
                level=taxonomy_tree.leaf_level,
                node=leaf)
        for leaf in taxonomy_tree.leaf_nodes
    }

    child_to_idx = {
        child: idx
        for idx, child in enumerate(data['reference_data'].cell_identifiers)
    }

    child_to_parent = {
        child_to_idx[child]: child_to_parent[child]
        for child in child_to_parent
    }

    for ii in range(bootstrap_iteration):
        _hann_iteration(
            query_cell_by_gene=data['query_data'],
            reference_cell_by_gene=data['reference_data'],
            taxonomy_tree=taxonomy_tree,
            as_leaves=taxonomy_tree.as_leaves,
            leaf_to_parent=child_to_parent,
            leaf_to_idx=child_to_idx,
            marker_lookup=data['marker_lookup'],
            bootstrap_factor=bootstrap_factor,
            rng=rng,
            votes_out=votes,
            corr_out=corr
        )


def _hann_iteration(
        query_cell_by_gene,
        reference_cell_by_gene,
        taxonomy_tree,
        as_leaves,
        leaf_to_parent,
        leaf_to_idx,
        marker_lookup,
        bootstrap_factor,
        rng,
        votes_out,
        corr_out):

    marker_idx = marker_lookup['None']
    query_subset = query_cell_by_gene.data[:, marker_idx]
    reference_subset = reference_cell_by_gene.data[:, marker_idx]

    chosen = correlation_nearest_neighbors(
        baseline_array=reference,
        query_array=query,
        return_correlation=False
    )

    cell_assignments = np.array(
        [leaf_to_parent[idx][taxonomy_tree.hierarchy[0]]
         for idx in chosen]
    )

    new_cell_assignments = np.array(['']*len(cell_assignments))

    for level in taxonomy_tree.hierarchy[1:-1]:
        unq_parents = np.unique(cell_assignments)
        for parent in unq_parents:
            cell_idx = np.where(cell_assignments==parent)[0]
            children = taxonomy_tree.children(level=level, node=parent)
            if len(children) == 1:
                new_cell_assignments[cell_idx] = children[0]
            else:
                leaf_idx = np.sort(
                    [leaf_to_idx[leaf] for leaf in as_leaves[level][parent]]
                )
                marker_idx = marker_idx[f'{level}/{parent}']

                query_subset = query_cell_by_gene.data[cell_idx, :]
                query_subset = query_subset[:, marker_idx]

                reference_subset = reference_cell_by_gene.data[leaf_idx, :]
                reference_subset = reference_subset[:, marker_idx]

                (chosen,
                 chosen_corr) = correlation_nearest_neighbors(
                                     baseline_array=reference_subset,
                                     query_array=query_subset,
                                     return_correlation=True)

                chosen = leaf_idx[chosen]
                if level == taxonomy_tree.leaf_level:
                    assignments = np.array(
                        [reference_cell_by_gene.cell_identifiers[idx]
                         for idx in chosen]
                    )
                else:
                    assignments = np.array(
                        [leaf_to_parent[idx][level] for idx in chosen]
                    )
                new_cell_assignments[cell_idx] = assignments

        cell_assignments = new_cell_assignments
