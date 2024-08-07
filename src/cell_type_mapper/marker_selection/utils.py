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

    up_markers = marker_gene_array.up_by_pair
    down_markers = marker_gene_array.down_by_pair

    for pair0 in range(0, n_taxon, batch_size):
        pair1 = min(n_pairs, pair0+batch_size)
        if taxonomy_mask is None:
            pair_batch = np.arange(pair0, pair1)
        else:
            pair_batch = taxonomy_mask[pair0:pair1]

        utility_sum += marker_gene_array.up_mask_from_pair_idx_batch(
            pair_batch)
        utility_sum += marker_gene_array.down_mask_from_pair_idx_batch(
            pair_batch)

        idx0 = down_markers.indptr[pair_batch]
        idx1 = down_markers.indptr[pair_batch+1]
        marker_census[pair0:pair1, 0] += idx1-idx0

        idx0 = up_markers.indptr[pair_batch]
        idx1 = up_markers.indptr[pair_batch+1]
        marker_census[pair0:pair1, 1] += idx1-idx0

    return utility_sum, marker_census
