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

    (child_to_idx,
     child_to_parent) = _get_lookups(
                             taxonomy_tree=taxonomy_tree,
                             reference_data=data['reference_data'])

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
        corr_out,
        min_chosen_markers=5):

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
    correlation_vector = np.zeros(len(cell_assignments), dtype=float)

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
                to_choose = max(
                    min_chosen_markers,
                    np.round(bootstrap_factor*len(marker_idx))
                )

                if to_choose < len(marker_idx):
                    marker_idx = np.sort(
                         rng.choice(marker_idx, to_choose, replace=False)
                    )

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
                    correlation_vector[cell_idx] = chosen_corr
                else:
                    assignments = np.array(
                        [leaf_to_parent[idx][level] for idx in chosen]
                    )
                new_cell_assignments[cell_idx] = assignments

        cell_assignments = new_cell_assignments

    for i_cell in range(len(cell_assignments)):
        idx = leaf_to_idx[cell_assignments[i_cell]]
        votes_out[i_cell, idx] += 1
        corr_out[i_cell, idx] += corrleation_vector[i_cell]


def _get_lookups(
        taxonomy_tree,
        reference_data):
    """
    Construct dicts mapping leaf nodes to their parents in a taxonomy
    tree and leaf nodes to their integer idx in a reference CellByGeneMatrix

    Parameters
    ----------
    taxonomy_tree:
        a TaxomomyTree

    reference_data:
        a CellByGeneMatrix

    Returns
    -------
    child_to_idx:
        dict mapping leaf nodes in the taxonomy tree to their
        row idx in refernence_data

    child_to_parent:
        dict mapping leaf nodes (as idx) in the taxonomy to
        parent nodes at all levels
    """
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

    return child_to_idx, child_to_parent
