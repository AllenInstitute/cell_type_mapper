import h5py
import json
import numpy as np

from hierarchical_mapping.diff_exp.scores import (
    aggregate_stats,
    read_precomputed_stats)

from hierarchical_mapping.cell_by_gene.cell_by_gene import (
    CellByGeneMatrix)


def assemble_query_data(
        full_query_data,
        mean_profile_matrix,
        taxonomy_tree,
        marker_cache_path,
        parent_node):
    """
    Assemble all of the data needed to select a taxonomy node
    for a collection of cells.

    Parameters
    ----------
    full_query_data:
        A CellByGeneMatrix containing the query data.
        Must have normalization == 'log2CPM'.
    mean_profile_matrix:
        A CellByGeneMatrix containing the mean gene expression profiles
        for each cell cluster in the reference taxonomy.
    taxonomy_tree:
        instance of
        hierarchical_mapping.taxonomty.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree
    marker_cache_path:
        Path to the HDF5 file recording the marker genes to be used
        for this (reference data, query data) pair
    parent_node:
        Either None (if we are querying from root) or a
        (parent_level, parent_node) tuple indicating the parent node
        in the taxonomy whose children we are choosing between.

    Returns
    --------
    A dict
        'query_data' -> a CellByGeneMatrix containing the query data
        downsampled to just the needed marker genes

        'reference_data' -> A CellByGeneMatrix containing mean_profile_matrix
        downsampled to the relevant clusters and marker genes

        'reference_types' -> A list indicating the taxonomic types implied
        by teh clusters in 'reference_data' (in the order they appear in
        'reference_data's rows)
    """

    tree_as_leaves = taxonomy_tree.as_leaves
    hierarchy = taxonomy_tree.hierarchy
    level_to_idx = {level: idx for idx, level in enumerate(hierarchy)}

    if parent_node is None:
        parent_grp = 'None'
        immediate_children = taxonomy_tree.nodes_at_level(hierarchy[0])
        child_level = hierarchy[0]

    else:
        parent_grp = f"{parent_node[0]}/{parent_node[1]}"

        immediate_children = taxonomy_tree.children(
               level=parent_node[0],
               node=parent_node[1])

        child_level = hierarchy[level_to_idx[parent_node[0]]+1]

    immediate_children.sort()

    leaf_to_type = dict()
    for child in immediate_children:
        for leaf in tree_as_leaves[child_level][child]:
            leaf_to_type[leaf] = child

    with h5py.File(marker_cache_path, 'r', swmr=True) as in_file:
        reference_markers = in_file[parent_grp]['reference'][()]
        raw_query_markers = in_file[parent_grp]['query'][()]
        all_ref_identifiers = json.loads(
            in_file["reference_gene_names"][()].decode("utf-8"))
        all_query_identifiers = json.loads(
            in_file["query_gene_names"][()].decode("utf-8"))

    # select only the desired query marker genes

    query_markers = [all_query_identifiers[ii]
                     for ii in raw_query_markers]

    query_data = full_query_data.downsample_genes(
        selected_genes=query_markers)

    reference_marker_identifiers = [
        all_ref_identifiers[ii] for ii in reference_markers]

    children = list(leaf_to_type.keys())
    children.sort()

    reference_data = mean_profile_matrix.downsample_cells(
        selected_cells=children)

    reference_data.downsample_genes_in_place(
        selected_genes=reference_marker_identifiers)

    reference_types = []
    for ii, child in enumerate(reference_data.cell_identifiers):
        reference_types.append(leaf_to_type[child])

    if query_data.gene_identifiers != reference_data.gene_identifiers:
        raise RuntimeError(
            "Mismatch between query marker genes and reference marker genes")

    if query_data.normalization != "log2CPM":
        raise RuntimeError(
            f"query data normalization is '{query_data.normalization}'\n"
            "should be 'log2CPM'")

    if reference_data.normalization != "log2CPM":
        raise RuntimeError(
            "reference data normalization is "
            f"'{reference_data.normalization}'\n"
            "should be 'log2CPM'")

    return {'query_data': query_data,
            'reference_data': reference_data,
            'reference_types': reference_types}


def get_leaf_means(
        taxonomy_tree,
        precompute_path):
    """
    Returns a CellByGeneMatrix in which each cell
    is a cluster and the .data array contains
    the mean gene expression profile of the cluster.
    """
    precomputed_stats = read_precomputed_stats(precompute_path)
    leaf_names = taxonomy_tree.all_leaves
    leaf_names.sort()
    result = dict()

    n_cells = len(leaf_names)
    data = None

    for i_leaf, leaf in enumerate(leaf_names):
        # gt1/0 threshold do not actually matter here
        stats = aggregate_stats(
                    leaf_population=[leaf,],
                    precomputed_stats=precomputed_stats['cluster_stats'])
        this_mean = stats['mean']
        if data is None:
            n_genes = len(this_mean)
            data = np.zeros((n_cells, n_genes))
        data[i_leaf, :] = this_mean

    result = CellByGeneMatrix(
        data=data,
        gene_identifiers=precomputed_stats['gene_names'],
        cell_identifiers=leaf_names,
        normalization="log2CPM")

    return result
