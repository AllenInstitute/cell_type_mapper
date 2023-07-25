import numpy as np


def create_utility_array(
        marker_gene_array,
        gb_size=10,
        taxonomy_mask=None):
    """
    Create an (n_genes,) array of how useful each gene is as a marker.
    Utility is just a count of how many (+/-, taxonomy_pair) combinations
    the gene is a marker for (in this case +/- indicates which node in the
    taxonomy pair the gene is up-regulated for).

    Parameters
    ----------
    marker_gene_array:
        A MarkerGeneArray
    gb_size:
        Number of gigabytes to load at a time (approximately)
    taxonomy_mask:
        if not None, a list of integers denoting which columns to
        sum utility for.

    Returns
    -------
    utility_arry:
        A numpy array of floats indicating the utility of each gene.

    marker_census:
        A numpy of ints indicating how many markers there are for
        each (taxonomy pair, sign) combination.

    Notes
    -----
    As implemented, it is assumed that the rows of the arrays in cache_path
    are genes and the columns are taxonomy pairs
    """

    if marker_gene_array.has_sparse:
        return create_utility_array_sparse(
            marker_gene_array=marker_gene_array,
            gb_size=gb_size,
            taxonomy_mask=taxonomy_mask)

    return create_utility_array_dense(
        marker_gene_array=marker_gene_array,
        gb_size=gb_size,
        taxonomy_mask=taxonomy_mask)


def create_utility_array_dense(
        marker_gene_array,
        gb_size=10,
        taxonomy_mask=None):
    """
    Create an (n_genes,) array of how useful each gene is as a marker.
    Utility is just a count of how many (+/-, taxonomy_pair) combinations
    the gene is a marker for (in this case +/- indicates which node in the
    taxonomy pair the gene is up-regulated for).

    This function uses the raw binary arrays in the marker array (as
    opposed to the sparse_by_pair arrays)

    Parameters
    ----------
    marker_gene_array:
        A MarkerGeneArray
    gb_size:
        Number of gigabytes to load at a time (approximately)
    taxonomy_mask:
        if not None, a list of integers denoting which columns to
        sum utility for.

    Returns
    -------
    utility_arry:
        A numpy array of floats indicating the utility of each gene.

    marker_census:
        A numpy of ints indicating how many markers there are for
        each (taxonomy pair, sign) combination.

    Notes
    -----
    As implemented, it is assumed that the rows of the arrays in cache_path
    are genes and the columns are taxonomy pairs
    """

    n_pairs = marker_gene_array.n_pairs
    n_genes = marker_gene_array.n_genes

    if taxonomy_mask is None:
        n_taxon = n_pairs
    else:
        n_taxon = len(taxonomy_mask)
    marker_census = np.zeros((n_taxon, 2), dtype=int)
    utility_sum = np.zeros(n_genes, dtype=int)

    byte_size = gb_size*1024**3
    batch_size = max(1, np.round(byte_size/(3*n_pairs)).astype(int))

    for gene0 in range(0, n_genes, batch_size):
        gene1 = min(n_genes, gene0+batch_size)
        up_reg_batch = marker_gene_array.up_regulated_gene_batch(gene0, gene1)
        marker_batch = marker_gene_array.is_marker_gene_batch(gene0, gene1)

        if taxonomy_mask is not None:
            marker_batch = marker_batch[:, taxonomy_mask]
            up_reg_batch = up_reg_batch[:, taxonomy_mask]

        utility_sum[gene0:gene1] = marker_batch.sum(axis=1)
        marker_census[:, 0] += (np.logical_not(up_reg_batch)
                                * marker_batch).sum(axis=0)
        marker_census[:, 1] += (up_reg_batch*marker_batch).sum(axis=0)

    return utility_sum, marker_census


def create_utility_array_sparse(
        marker_gene_array,
        gb_size=10,
        taxonomy_mask=None):
    """
    Create an (n_genes,) array of how useful each gene is as a marker.
    Utility is just a count of how many (+/-, taxonomy_pair) combinations
    the gene is a marker for (in this case +/- indicates which node in the
    taxonomy pair the gene is up-regulated for).

    This function uses the sparse arrays in the marker_gene_array

    Parameters
    ----------
    marker_gene_array:
        A MarkerGeneArray
    gb_size:
        Number of gigabytes to load at a time (approximately)
    taxonomy_mask:
        if not None, a list of integers denoting which columns to
        sum utility for.

    Returns
    -------
    utility_arry:
        A numpy array of floats indicating the utility of each gene.

    marker_census:
        A numpy of ints indicating how many markers there are for
        each (taxonomy pair, sign) combination.

    Notes
    -----
    As implemented, it is assumed that the rows of the arrays in cache_path
    are genes and the columns are taxonomy pairs
    """

    n_pairs = marker_gene_array.n_pairs
    n_genes = marker_gene_array.n_genes

    if taxonomy_mask is None:
        n_taxon = n_pairs
    else:
        n_taxon = len(taxonomy_mask)

    marker_census = np.zeros((n_taxon, 2), dtype=int)
    utility_sum = np.zeros(n_genes, dtype=int)

    byte_size = gb_size*1024**3
    batch_size = max(1, np.round(byte_size/(3*n_genes)).astype(int))

    up_markers = marker_gene_array._up_marker_sparse_by_pair
    down_markers = marker_gene_array._down_marker_sparse_by_pair

    for pair0 in range(0, n_taxon, batch_size):
        pair1 = min(n_pairs, pair0+batch_size)
        if taxonomy_mask is None:
            pair_idx = np.arange(pair0, pair1)
        else:
            pair_idx = taxonomy_mask[pair0:pair1]

        (up_pairs,
         up_genes) = up_markers.get_sparse_genes_for_pair_array(
            pair_idx)
        (down_pairs,
         down_genes) = down_markers.get_sparse_genes_for_pair_array(
             pair_idx)

        up_idx, up_ct = np.unique(up_genes, return_counts=True)
        utility_sum[up_idx] += up_ct
        down_idx, down_ct = np.unique(down_genes, return_counts=True)
        utility_sum[down_idx] += down_ct

        marker_census[pair0:pair1, 0] += down_pairs[1:]-down_pairs[:-1]
        marker_census[pair0:pair1, 1] += up_pairs[1:]-up_pairs[:-1]

    return utility_sum, marker_census
