import h5py
import numpy as np

import cell_type_mapper.utils.utils as mapper_utils
import cell_type_mapper.utils.distance_utils as distance_utils
import cell_type_mapper.type_assignment.matching as matching


def hann_tally_votes(
        full_query_data,
        leaf_node_matrix,
        marker_gene_cache_path,
        taxonomy_tree,
        bootstrap_factor_lookup,
        bootstrap_iteration,
        rng):
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

    for ii in range(bootstrap_iteration):
        _hann_iteration(
            query_cell_by_gene=data['query_data'],
            reference_cell_by_gene=data['reference_data'],
            taxonomy_tree=taxonomy_tree,
            marker_lookup=data['marker_lookup'],
            bootstrap_factor_lookup=bootstrap_factor_lookup,
            rng=rng,
            votes_out=votes,
            corr_out=corr
        )

    return {
        'votes': votes,
        'correlation_sum': corr,
        'cell_identifiers': full_query_data.cell_identifiers
    }


def _hann_iteration(
        query_cell_by_gene,
        reference_cell_by_gene,
        taxonomy_tree,
        marker_lookup,
        bootstrap_factor_lookup,
        rng,
        votes_out,
        corr_out,
        min_chosen_markers=5):

    cell_assignments = np.array(
        ['']*query_cell_by_gene.n_cells
    ).astype(object)

    new_cell_assignments = np.array(
        ['']*query_cell_by_gene.n_cells
    ).astype(object)

    correlation_vector = np.zeros(
        query_cell_by_gene.n_cells,
        dtype=float
    )

    _assign_children_of_one_parent(
        cell_assignments=cell_assignments,
        new_cell_assignments=new_cell_assignments,
        correlation_vector=correlation_vector,
        taxonomy_tree=taxonomy_tree,
        parent_level=None,
        parent=None,
        marker_lookup=marker_lookup,
        rng=rng,
        bootstrap_factor=bootstrap_factor_lookup['None'],
        min_chosen_markers=min_chosen_markers,
        query_cell_by_gene=query_cell_by_gene,
        reference_cell_by_gene=reference_cell_by_gene)

    cell_assignments[:] = new_cell_assignments

    for level in taxonomy_tree.hierarchy[:-1]:
        unq_parents = np.unique(cell_assignments)
        for parent in unq_parents:
            _assign_children_of_one_parent(
                cell_assignments=cell_assignments,
                new_cell_assignments=new_cell_assignments,
                correlation_vector=correlation_vector,
                taxonomy_tree=taxonomy_tree,
                parent_level=level,
                parent=parent,
                marker_lookup=marker_lookup,
                rng=rng,
                bootstrap_factor=bootstrap_factor_lookup[level],
                min_chosen_markers=min_chosen_markers,
                query_cell_by_gene=query_cell_by_gene,
                reference_cell_by_gene=reference_cell_by_gene)

        cell_assignments[:] = new_cell_assignments

    _update_hann_votes(
        cell_assignments=cell_assignments,
        correlation_vector=correlation_vector,
        reference_cell_by_gene=reference_cell_by_gene,
        votes_out=votes_out,
        corr_out=corr_out
    )


def _assign_children_of_one_parent(
        cell_assignments,
        new_cell_assignments,
        correlation_vector,
        taxonomy_tree,
        parent_level,
        parent,
        marker_lookup,
        rng,
        bootstrap_factor,
        min_chosen_markers,
        query_cell_by_gene,
        reference_cell_by_gene):

    if parent is not None:
        cell_idx = np.where(cell_assignments == parent)[0]
    else:
        cell_idx = np.arange(len(cell_assignments), dtype=int)

    children = taxonomy_tree.children(level=parent_level, node=parent)

    if len(children) == 1:
        new_cell_assignments[cell_idx] = children[0]
    else:

        if parent is not None:
            marker_idx = marker_lookup[f'{parent_level}/{parent}']
        else:
            marker_idx = marker_lookup['None']

        to_choose = max(
            min_chosen_markers,
            np.round(bootstrap_factor*len(marker_idx)).astype(int)
        )

        if to_choose < len(marker_idx):
            marker_idx = np.sort(
                 rng.choice(marker_idx, to_choose, replace=False)
            )

        query_subset = query_cell_by_gene.data[cell_idx, :]
        query_subset = query_subset[:, marker_idx]

        reference_subset = reference_cell_by_gene.data[:, marker_idx]
        if parent is not None:
            leaf_idx = np.sort(
                [reference_cell_by_gene.cell_to_row[leaf]
                 for leaf in taxonomy_tree.as_leaves[parent_level][parent]]
            )

            reference_subset = reference_subset[leaf_idx, :]

        (chosen,
         chosen_corr) = distance_utils.correlation_nearest_neighbors(
                             baseline_array=reference_subset,
                             query_array=query_subset,
                             return_correlation=True)

        if parent is not None:
            chosen = leaf_idx[chosen]

        if parent_level == taxonomy_tree.hierarchy[-2]:
            assignments = np.array(
                [reference_cell_by_gene.cell_identifiers[idx]
                 for idx in chosen]
            )
            correlation_vector[cell_idx] = chosen_corr
        else:
            assignments = np.array(
                [taxonomy_tree.parents(
                    level=taxonomy_tree.leaf_level,
                    node=reference_cell_by_gene.cell_identifiers[idx])[
                        taxonomy_tree.child_level(parent_level)
                    ]
                 for idx in chosen
                 ]
            )
        new_cell_assignments[cell_idx] = assignments


def _update_hann_votes(
        cell_assignments,
        correlation_vector,
        reference_cell_by_gene,
        votes_out,
        corr_out):

    for i_cell in range(len(cell_assignments)):
        idx = reference_cell_by_gene.cell_to_row[cell_assignments[i_cell]]
        votes_out[i_cell, idx] += 1
        corr_out[i_cell, idx] += correlation_vector[i_cell]


def save_results(results, results_output_path):
    with h5py.File(results_output_path, "w") as dst:
        dst.create_dataset(
            "votes",
            data=results["votes"]
        )

        denom = np.where(
            results["votes"] > 0,
            results["votes"],
            1
        ).astype(float)

        corr = results["correlation_sum"]/denom

        dst.create_dataset(
            "correlation",
            data=corr
        )

        dst.create_dataset(
            "cell_identifiers",
            data=results["cell_identifiers"]
        )
