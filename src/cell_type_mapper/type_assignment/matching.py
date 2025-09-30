import h5py
import json
import numpy as np

from cell_type_mapper.diff_exp.score_utils import (
    read_precomputed_stats)

from cell_type_mapper.cell_by_gene.cell_by_gene import (
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
        cell_type_mapper.taxonomy.taxonomy_tree.TaxonomyTree
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
        by the clusters in 'reference_data' (in the order they appear in
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

    children = list(leaf_to_type.keys())
    children.sort()

    with h5py.File(marker_cache_path, 'r', swmr=True) as in_file:
        if parent_grp not in in_file:
            raise RuntimeError(
                f"{parent_grp} not in marker cache path ({marker_cache_path})")

        this_grp = in_file[parent_grp]

        for k in ("reference", "query"):
            if k not in this_grp:
                raise RuntimeError(
                    f"'{k}' not in group '{parent_grp}' of marker cache path")

        reference_markers = this_grp['reference'][()]
        all_ref_identifiers = json.loads(
            in_file["reference_gene_names"][()].decode("utf-8"))

    desired_genes = [
        all_ref_identifiers[ii] for ii in reference_markers
    ]

    # select only the desired query marker genes
    result = subset_cell_by_gene(
        full_query_data=full_query_data,
        mean_profile_matrix=mean_profile_matrix,
        desired_clusters=children,
        desired_genes=desired_genes
    )

    reference_types = []
    for ii, child in enumerate(result['reference_data'].cell_identifiers):
        reference_types.append(leaf_to_type[child])

    result['reference_types'] = reference_types
    return result


def assemble_query_data_hann(
        full_query_data,
        mean_profile_matrix,
        taxonomy_tree,
        marker_cache_path):
    """
    Assemble all of the data needed to select a taxonomy node
    for a collection of cells via the HANN algorithm.

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
        cell_type_mapper.taxonomy.taxonomy_tree.TaxonomyTree
        ecoding the taxonomy tree
    marker_cache_path:
        Path to the HDF5 file recording the marker genes to be used
        for this (reference data, query data) pair

    Returns
    --------
    A dict
        'query_data' -> a CellByGeneMatrix containing the query data
        downsampled to just the needed marker genes

        'reference_data' -> A CellByGeneMatrix containing mean_profile_matrix
        downsampled to the relevant clusters and marker genes

        'marker_lookup' -> a dict mapping taxons to marker gene arrays
        as indexes of the gene_identifiers in query_data and reference_data
    """

    desired_clusters = taxonomy_tree.nodes_at_level(
        taxonomy_tree.leaf_level
    )

    parent_nodes = ['None']
    for level in taxonomy_tree.hierarchy[:-1]:
        for node in taxonomy_tree.nodes_at_level(level):
            if len(taxonomy_tree.children(level=level, node=node)) > 1:
                parent_nodes.append(f'{level}/{node}')

    marker_lookup = dict()
    desired_genes = set()
    with h5py.File(marker_cache_path, 'r', swmr=True) as in_file:
        all_ref_identifiers = json.loads(
            in_file["reference_gene_names"][()].decode("utf-8"))
        for parent in parent_nodes:
            this = [all_ref_identifiers[ii]
                    for ii in in_file[parent]['reference'][()]]
            marker_lookup[parent] = this
            desired_genes = desired_genes.union(set(this))

    desired_genes = np.sort(list(desired_genes))

    result = subset_cell_by_gene(
        full_query_data=full_query_data,
        mean_profile_matrix=mean_profile_matrix,
        desired_clusters=desired_clusters,
        desired_genes=desired_genes
    )

    gene_to_idx = {
        gene: idx
        for idx, gene in enumerate(result['reference_data'].gene_identifiers)
    }

    marker_as_idx = dict()
    for key in marker_lookup:
        marker_as_idx[key] = np.sort([
            gene_to_idx[gene] for gene in marker_lookup[key]
        ])

    result['marker_lookup'] = marker_as_idx
    return result


def subset_cell_by_gene(
        full_query_data,
        mean_profile_matrix,
        desired_clusters,
        desired_genes):
    """
    Subset cell-by-gene-matrices so that they have the same marker genes

    Parameters
    ----------
    full_query_data:
        CellByGeneMatrix of the full query data
    mean_profile_matrix:
        CellByGeneMatrix of the full reference data
    desired_clusters:
        list of leaf nodes in mean_profile_matrix that we want
    desired_genes:
        list of gene identifiers we are downsampling to

    Returns
    -------
    A dict
        'query_data' -> a CellByGeneMatrix containing the query data
        downsampled to just the needed marker genes

        'reference_data' -> A CellByGeneMatrix containing mean_profile_matrix
        downsampled to the relevant clusters and marker genes
    """

    query_data = full_query_data.downsample_genes(
        selected_genes=desired_genes)

    reference_data = mean_profile_matrix.downsample_cells_by_name(
        selected_cells=desired_clusters)

    reference_data.downsample_genes_in_place(
        selected_genes=desired_genes)

    if not np.array_equal(
                np.array(query_data.gene_identifiers),
                np.array(reference_data.gene_identifiers)):
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
            'reference_data': reference_data}


def get_leaf_means(
        taxonomy_tree,
        precompute_path,
        for_marker_selection=True):
    """
    Returns a CellByGeneMatrix in which each cell
    is a cluster and the .data array contains
    the mean gene expression profile of the cluster.

    If for_marker_selection is True and 'sumsq' or 'ge1' are missing,
    raise an error
    """
    precomputed_stats = read_precomputed_stats(
        precomputed_stats_path=precompute_path,
        taxonomy_tree=taxonomy_tree,
        for_marker_selection=for_marker_selection)
    leaf_names = taxonomy_tree.all_leaves
    leaf_names.sort()
    result = dict()

    n_cells = len(leaf_names)
    data = None

    for i_leaf, leaf in enumerate(leaf_names):
        # gt1/0 threshold do not actually matter here
        leaf_key = f'{taxonomy_tree.leaf_level}/{leaf}'
        stats = precomputed_stats['cluster_stats'][leaf_key]
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
