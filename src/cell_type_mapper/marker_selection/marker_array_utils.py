import numpy as np

from cell_type_mapper.corr.utils import (
    match_genes)


def thin_marker_gene_array_by_gene(
        marker_gene_array,
        query_gene_names,
        tmp_dir=None):
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
    tmp_dir:
        Directory for storing scratch files so big matrix manipulations
        do not happen in memory.

    Returns
    -------
    marker_gene_array:
        With only the nodes that overlap with query_gene_naems
        returned.

    Note
    -----
    This method alters marker_gene_array in place
    """

    reference_gene_mask = query_genes_to_mask(
        reference_gene_names=marker_gene_array.gene_names,
        query_gene_names=query_gene_names)

    if reference_gene_mask.sum() == marker_gene_array.n_genes:
        # nothing to be done; query and reference genes are
        # the same
        return marker_gene_array

    reference_gene_idx = np.where(reference_gene_mask)[0]
    marker_gene_array.downsample_genes(
            reference_gene_idx,
            tmp_dir=tmp_dir)
    return marker_gene_array


def query_genes_to_mask(
        reference_gene_names,
        query_gene_names):
    """
    Return a mask indicating which genes in reference_gene_names
    need to be kept to align with query_gene_names
    """
    # figure out which genes are in both the reference dataset
    # and the query dataset
    matched_genes = match_genes(
        reference_gene_names=reference_gene_names,
        query_gene_names=query_gene_names)

    if len(matched_genes['reference']) == 0:
        raise RuntimeError(
            "No gene overlap between reference and query set")

    reference_gene_mask = np.zeros(len(reference_gene_names), dtype=bool)
    reference_gene_mask[matched_genes['reference']] = True
    return reference_gene_mask


def _create_new_pair_lookup(only_keep_pairs):
    """
    Create new pair-to-idx lookup for case where we
    are only keeping the specified pairs
    """
    new_lookup = dict()
    for ii, pair in enumerate(only_keep_pairs):
        level = pair[0]
        node1 = pair[1]
        node2 = pair[2]
        if level not in new_lookup:
            new_lookup[level] = dict()
        if node1 not in new_lookup[level]:
            new_lookup[level][node1] = dict()
        new_lookup[level][node1][node2] = ii
    return new_lookup


def _idx_of_pair(
        taxonomy_pair_to_idx,
        level,
        node1,
        node2):
    if node1 not in taxonomy_pair_to_idx[level]:
        raise RuntimeError(
            f"{node1} not under taxonomy level {level}")
    if node2 not in taxonomy_pair_to_idx[level][node1]:
        raise RuntimeError(
            f"({level},  {node1}, {node2})\n"
            "not a valid taxonomy pair specification; try reversing "
            "node1 and node2")

    pair_idx = taxonomy_pair_to_idx[level][node1][node2]
    return pair_idx
